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

class SpeakerEmbedding(nn.Module):
    def __init__(self,
                 spec_channels=513,
                 inter_channels=512,
                 hidden_channels=512,
                 speaker_emb_dim=512,
                 ckpt_path=None,
                 mode='inference'):
        super().__init__()

        # 线性谱
        self.hop_length = 300
        self.win_length = 1024
        self.filter_length = 1024
        self.sampling_rate = 24000

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
        # use mel feature to extract initial style embedding.
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
        else:
            self.freeze_BN = True   # BatchNorm 在使用的时候一定要注意设成eval模式
        if self.freeze_BN:
            logger.info(f"SpeakerEmbeding Mudule Freeze the BN layer")
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
        fbank = torch.stack(fbanks)  # B D T
        # 音色编码由fbank得到
        timbre_vec = self.speaker_encoder(fbank).unsqueeze(-1)  # B D 1

        melspec = spectrogram_torch(wave.squeeze(1),     # B 1 T -> B T
                                    self.filter_length,
                                    self.sampling_rate,
                                    self.hop_length,
                                    self.win_length)  # B D T

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