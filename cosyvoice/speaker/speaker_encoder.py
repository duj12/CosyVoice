# Copyright (c) 2024  Jing Du  (thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import logging
import os
from torch import nn
import torchaudio.functional as AF
from cosyvoice.speaker.CAMPPlus import CAMPPlus, FBank
from cosyvoice.speaker.mel_processing import spectrogram_torch
from cosyvoice.speaker.posterior_encoder import PosteriorEncoder
from cosyvoice.speaker.style_encoder import (
    StyleFuser, StyleEncoder_v2, GlobalStyleTokens_v2
)
import cosyvoice.speaker.commons as commons

logger = logging.getLogger(__name__)


def freeze_BN_layer(m):
    cls = m.__class__.__name__
    if "BatchNorm" in cls:
        m.eval()

class SpecAugment(torch.nn.Module):
    r"""Apply time and frequency masking to a spectrogram. From torchaudio
    Args:
        n_time_masks (int): Number of time masks. If its value is zero, no time masking will be applied.
        time_mask_ratio (float): Maximum possible ratio of the time mask in time dimension.
        n_freq_masks (int): Number of frequency masks. If its value is zero, no frequency masking will be applied.
        freq_mask_ratio (float): Maximum possible ratio of the frequency mask in frequency dimension.
        iid_masks (bool, optional): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 3D or more. (Default: ``True``)
        p (float, optional): maximum proportion of time steps that can be masked.
            Must be within range [0.0, 1.0]. (Default: 1.0)
        zero_masking (bool, optional): If ``True``, use 0 as the mask value,
            else use mean of the input tensor. (Default: ``False``)
    """
    __constants__ = [
        "n_time_masks",
        "time_mask_ratio",
        "n_freq_masks",
        "freq_mask_ratio",
        "iid_masks",
        "p",
        "zero_masking",
    ]

    def __init__(
        self,
        n_time_masks: int,
        time_mask_ratio: float,
        n_freq_masks: int,
        freq_mask_ratio: float,
        iid_masks: bool = True,
        p: float = 1.0,
        zero_masking: bool = False,
    ) -> None:
        super(SpecAugment, self).__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_ratio = time_mask_ratio
        self.n_freq_masks = n_freq_masks
        self.freq_mask_ratio = freq_mask_ratio
        self.iid_masks = iid_masks
        self.p = p
        self.zero_masking = zero_masking

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of shape `(..., freq, time)`.
        Returns:
            Tensor: Masked spectrogram of shape `(..., freq, time)`.
        """
        if self.zero_masking:
            mask_value = 0.0
        else:
            mask_value = specgram.mean()

        if specgram.dim() == 3:
            specgram = specgram.unsqueeze(1)  # add 1 dim, to (B, C, D, T)

        time_dim = specgram.dim() - 1
        freq_dim = time_dim - 1
        time_size = specgram.size(time_dim)
        freq_size = specgram.size(freq_dim)
        time_mask_param = int(time_size * self.time_mask_ratio)
        freq_mask_param = int(freq_size * self.freq_mask_ratio)

        if specgram.dim() > 2 and self.iid_masks is True:
            for _ in range(self.n_time_masks):
                specgram = AF.mask_along_axis_iid(specgram, time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = AF.mask_along_axis_iid(specgram, freq_mask_param, mask_value, freq_dim, p=self.p)
        else:
            for _ in range(self.n_time_masks):
                specgram = AF.mask_along_axis(specgram, self.time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = AF.mask_along_axis(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p)

        if specgram.dim() == 4:
            specgram = specgram.squeeze(1)  # back to (B, D, T)

        return specgram


class SpeakerEmbedding(nn.Module):
    def __init__(
            self, spec_channels=513, inter_channels=512, hidden_channels=512,
            speaker_emb_dim=512, ckpt_path=None, mode='inference',
            freeze_post_enc=True, freeze_timbre_enc=True, freeze_style_enc=True,
            spk_audio_crop=10,  # 提取向量是音频裁剪最短长度，低于此时长音频会被拼接
            spec_aug_config=None,   # 计算对比损失时进行SpecAug {mask_ratio=0.1, iid_mask=True}
            spk_mix=False,      # 一个batch内音频进行拼接增强
            noise_aug_config=None,  # 参考音频加噪增强{'noise_list': None, 'db_range': [0, 20]}
    ):
        super().__init__()

        # linear spectrum, fixed configuration
        self.hop_length = 300
        self.win_length = 1024
        self.filter_length = 1024
        self.sampling_rate = 24000
        self.spk_audio_crop = spk_audio_crop
        self.spec_aug_config = spec_aug_config
        self.spk_mix = spk_mix
        self.noise_aug_config = noise_aug_config

        if self.spec_aug_config is not None and self.training:
            self.spec_aug = SpecAugment(
                n_time_masks=1, time_mask_ratio=spec_aug_config['mask_ratio'],
                n_freq_masks=1, freq_mask_ratio=spec_aug_config['mask_ratio'],
                iid_masks=spec_aug_config['iid_mask'],
            )

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels,
                                      hidden_channels, 5, 1, 16,
                                      gin_channels=0)

        # now we use fixed dimension of pretrained SV model.
        # support 80 dimension Fbank input, 192 dimension embedding output.
        fbank_dim = 80
        self.fbank_sr = 16000
        self.fbank_extractor = FBank(
            fbank_dim, sample_rate=self.fbank_sr, mean_nor=True)
        self.speaker_encoder = CAMPPlus(
            feat_dim=fbank_dim, embedding_size=192)

        # Style Encoding
        style_dim = speaker_emb_dim
        gst_vector_dim = speaker_emb_dim

        style_hidden = 256
        style_head = 4
        style_kernel_size = 5
        style_layers = 6
        # use post encoder feature to extract initial style embedding.
        self.style_encoder = StyleEncoder_v2(
            hidden_channels, style_hidden, style_dim,
            style_kernel_size, style_head, num_layers=style_layers)

        # Style Tokens Attention
        gst_token_num = 4096
        gst_att_head = 8
        gst_layers = 6
        self.gst = GlobalStyleTokens_v2(
            gst_token_num, gst_vector_dim, gst_att_head, num_layers=gst_layers)

        if 192 != style_dim:
            # convert speakerencoder output into n_speaker_dim
            self.speaker_adapter = StyleFuser(style_dim, 192, fuse_type="Add")
        else:
            self.speaker_adapter = None

        if mode == 'train':
            self.freeze_BN = False
            if freeze_post_enc:
                for param in self.enc_q.parameters():
                    param.requires_grad = False
                logger.info(f"Posterior Encoder do not execute back propagation.")

            if freeze_timbre_enc:
                for param in self.speaker_encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Timbre Encoder do not execute back propagation.")

            if freeze_style_enc:
                for param in self.style_encoder.parameters():
                    param.requires_grad = False
                for param in self.gst.parameters():
                    param.requires_grad = False
                if self.speaker_adapter is not None:
                    for param in self.speaker_adapter.parameters():
                        param.requires_grad = False
                logger.info(f"Style Encoder do not execute back propagation.")

        else:
            self.freeze_BN = True   # BatchNorm 在使用的时候一定要注意设成eval模式
        if self.freeze_BN:
            logger.info(f"SpeakerEmbedding Module Freeze the BN layer")
            for m in self.modules():
                freeze_BN_layer(m)

        if ckpt_path is not None:
            assert os.path.exists(ckpt_path), \
                f"ckpt path {ckpt_path} can not access."
            logger.info(f"loading pretrained speaker encoder {ckpt_path}.")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self,  wave, wave_lengths):
        B = wave.size(0)
        # wave B 1 T
        fbanks = []
        for i in range(B):
            fbank = self.fbank_extractor(
                wave[i], sr=self.sampling_rate).transpose(0, 1)  # D T
            fbanks.append(fbank)
        fbank = torch.stack(fbanks)  # B D
        melspec = spectrogram_torch(wave.squeeze(1),     # B 1 T -> B T
                                    self.filter_length,
                                    self.sampling_rate,
                                    self.hop_length,
                                    self.win_length)  # B D T

        if self.spec_aug_config is not None and self.training:
            fbank = fbank.unsqueeze(1)  # to (BCDT)
            fbank = self.spec_aug(fbank).squeeze(1)  # to BDT
            melspec = melspec.unsqueeze(1)   # to BCDT
            melspec = self.spec_aug(melspec).squeeze(1)  # back to BDT

        # 音色编码由fbank得到
        timbre_vec = self.speaker_encoder(fbank).unsqueeze(-1)  # B D 1

        total_length = melspec.size(-1)
        spec_lengths = wave_lengths // self.hop_length

        # 线性谱输入到后延编码器，得到隐层表征
        melspec, m_q, logs_q, y_mask = self.enc_q(melspec, spec_lengths, g=None)

        style_mask = commons.sequence_mask(
            spec_lengths, total_length).unsqueeze(1)  # [B,1,T]

        # 隐层表征输入风格编码器，得到风格编码
        style_vec = self.style_encoder(melspec.transpose(1, 2), style_mask)  # B D

        # 风格编码输入GST
        style_vec = self.gst(style_vec)  # B 1 D

        # 风格和音色融合
        if self.speaker_adapter is not None:
            g = self.speaker_adapter(
                style_vec, timbre_vec.transpose(1, 2)).transpose(1, 2)
        else:
            g = timbre_vec + style_vec.transpose(1, 2)  # B D 1

        g = g.squeeze(-1)  # B D 1 -> B D
        return g


class SpeakerEmbedding_wo_PostEnc(nn.Module):
    def __init__(
            self, spec_channels=513, hidden_channels=512,
            speaker_emb_dim=512, ckpt_path=None, mode='inference',
            freeze_timbre_enc=True, freeze_style_enc=True,
            spk_audio_crop=10,  # 提取向量是音频裁剪最短长度，低于此时长音频会被拼接
            spec_aug_config=None,   # 计算对比损失时进行SpecAug {mask_ratio=0.1, iid_mask=True}
            spk_mix=False,      # 一个batch内音频进行拼接增强
            noise_aug_config=None,  # 参考音频加噪增强{'noise_list': None, 'db_range': [0, 20]}
    ):
        super().__init__()

        # linear spectrum, fixed configuration
        self.hop_length = 300
        self.win_length = 1024
        self.filter_length = 1024
        self.sampling_rate = 24000
        self.spk_audio_crop = spk_audio_crop
        self.spec_aug_config = spec_aug_config
        self.spk_mix = spk_mix
        self.noise_aug_config = noise_aug_config

        if self.spec_aug_config is not None and self.training:
            self.spec_aug = SpecAugment(
                n_time_masks=1, time_mask_ratio=spec_aug_config['mask_ratio'],
                n_freq_masks=1, freq_mask_ratio=spec_aug_config['mask_ratio'],
                iid_masks=spec_aug_config['iid_mask'],
            )

        self.enc_q = nn.Linear(spec_channels, hidden_channels)

        # now we use fixed dimension of pretrained SV model.
        # support 80 dimension Fbank input, 192 dimension embedding output.
        fbank_dim = 80
        self.fbank_sr = 16000
        self.fbank_extractor = FBank(
            fbank_dim, sample_rate=self.fbank_sr, mean_nor=True)
        self.speaker_encoder = CAMPPlus(
            feat_dim=fbank_dim, embedding_size=192)

        # Style Encoding
        style_dim = speaker_emb_dim
        gst_vector_dim = speaker_emb_dim

        style_hidden = 256
        style_head = 4
        style_kernel_size = 5
        style_layers = 6
        # use post encoder feature to extract initial style embedding.
        self.style_encoder = StyleEncoder_v2(
            hidden_channels, style_hidden, style_dim,
            style_kernel_size, style_head, num_layers=style_layers)

        # Style Tokens Attention
        gst_token_num = 4096
        gst_att_head = 8
        gst_layers = 6
        self.gst = GlobalStyleTokens_v2(
            gst_token_num, gst_vector_dim, gst_att_head, num_layers=gst_layers)

        if 192 != style_dim:
            # convert speakerencoder output into n_speaker_dim
            self.speaker_adapter = StyleFuser(style_dim, 192, fuse_type="Add")
        else:
            self.speaker_adapter = None

        if mode == 'train':
            self.freeze_BN = False

            if freeze_timbre_enc:
                for param in self.speaker_encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Timbre Encoder do not execute back propagation.")

            if freeze_style_enc:
                for param in self.style_encoder.parameters():
                    param.requires_grad = False
                for param in self.gst.parameters():
                    param.requires_grad = False
                if self.speaker_adapter is not None:
                    for param in self.speaker_adapter.parameters():
                        param.requires_grad = False
                logger.info(f"Style Encoder do not execute back propagation.")

        else:
            self.freeze_BN = True   # BatchNorm 在使用的时候一定要注意设成eval模式
        if self.freeze_BN:
            logger.info(f"SpeakerEmbedding Module Freeze the BN layer")
            for m in self.modules():
                freeze_BN_layer(m)

        if ckpt_path is not None:
            assert os.path.exists(ckpt_path), \
                f"ckpt path {ckpt_path} can not access."
            logger.info(f"loading pretrained speaker encoder {ckpt_path}.")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def forward(self,  wave, wave_lengths):
        B = wave.size(0)
        # wave B 1 T
        fbanks = []
        for i in range(B):
            fbank = self.fbank_extractor(
                wave[i], sr=self.sampling_rate).transpose(0, 1)  # D T
            fbanks.append(fbank)
        fbank = torch.stack(fbanks)  # B D
        melspec = spectrogram_torch(wave.squeeze(1),     # B 1 T -> B T
                                    self.filter_length,
                                    self.sampling_rate,
                                    self.hop_length,
                                    self.win_length)  # B D T

        if self.spec_aug_config is not None and self.training:
            fbank = fbank.unsqueeze(1)  # to (BCDT)
            fbank = self.spec_aug(fbank).squeeze(1)  # to BDT
            melspec = melspec.unsqueeze(1)   # to BCDT
            melspec = self.spec_aug(melspec).squeeze(1)  # back to BDT

        # 音色编码由fbank得到
        timbre_vec = self.speaker_encoder(fbank).unsqueeze(-1)  # B D 1

        total_length = melspec.size(-1)
        spec_lengths = wave_lengths // self.hop_length

        # 线性谱维度变换
        melspec = self.enc_q(melspec.transpose(1,2)).transpose(1,2)

        style_mask = commons.sequence_mask(
            spec_lengths, total_length).unsqueeze(1)  # [B,1,T]

        # 隐层表征输入风格编码器，得到风格编码
        style_vec = self.style_encoder(melspec.transpose(1, 2), style_mask)  # B D

        # 风格编码输入GST
        style_vec = self.gst(style_vec)  # B 1 D

        # 风格和音色融合
        if self.speaker_adapter is not None:
            g = self.speaker_adapter(
                style_vec, timbre_vec.transpose(1, 2)).transpose(1, 2)
        else:
            g = timbre_vec + style_vec.transpose(1, 2)  # B D 1

        g = g.squeeze(-1)  # B D 1 -> B D
        return g