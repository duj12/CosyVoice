# Copyright (c) 2025 Jing Du
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
# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/bigvgan.py

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm
from cosyvoice.utils.mask import make_pad_mask
import cosyvoice.BigVGAN.nnet.activations as activations
from cosyvoice.BigVGAN.utils import get_padding, init_weights
from cosyvoice.BigVGAN.alias_free_activation.torch.act import Activation1d as TorchActivation1d
from cosyvoice.BigVGAN.alias_free_activation.cuda.activation1d import Activation1d as CudaActivation1d


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        snake_logscale=True,
        use_cuda_kernel: bool = False
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        snake_logscale=True,
        use_cuda_kernel: bool = False
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


'''
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
'''
class BigVGAN(
    torch.nn.Module,
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(
            self,
            vocab_size=6561,
            input_size=512,
            output_size=1024,
            mel_bin=80,
            resblock="1",
            upsample_rates=[4,4,4,4,2,2],
            upsample_kernel_sizes=[8,8,4,4,4,4],
            upsample_initial_channel=1536,
            resblock_kernel_sizes=[3,7,11],
            resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
            speaker_embedding_dim=512,
            cond_d_vector_in_each_upsampling_layer=True,
            activation="snakebeta",
            snake_logscale=True,
            use_cuda_kernel: bool=False,
            encoder1=None,
            encoder2=None,

    ):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        if self.encoder2 is not None:
            self.encoder_proj = torch.nn.Linear(self.encoder2.output_size(), output_size)
            self.mel_proj = torch.nn.Linear(self.encoder2.output_size(), mel_bin)
        else:
            self.encoder_proj = torch.nn.Linear(input_size, output_size)
            self.mel_proj = torch.nn.Linear(upsample_initial_channel, mel_bin)

        self.speaker_dim = speaker_embedding_dim
        self.use_cuda_kernel = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.use_cuda_kernel:
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.cond_in_each_up_layer = cond_d_vector_in_each_upsampling_layer

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(output_size, upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if resblock == "1":
            resblock_class = AMPBlock1
        elif resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(ch, k, d, activation=activation, snake_logscale=snake_logscale, use_cuda_kernel=use_cuda_kernel)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=snake_logscale)
            if activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=snake_logscale)
                if activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = True
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = True
        self.cond_layer = nn.Conv1d(speaker_embedding_dim, upsample_initial_channel, 1)
        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                self.conds.append(nn.Conv1d(speaker_embedding_dim, ch, 1))

    def forward(self, batch: dict,
                device: torch.device,):

        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        speaker_embedding = batch['embedding'].unsqueeze(-1).to(device)

        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        x = self.input_embedding(torch.clamp(token, min=0)) * mask

        # encoding and upsample speech token into feature
        if self.encoder1 is not None:
            x, _ = self.encoder1(x, token_len)   # upsample * 2
            token_len = token_len * 2
        if self.encoder2 is not None:
            x, _ = self.encoder2(x, token_len)       # upsample * 2
            token_len = token_len * 2
            mel_feat_out = self.mel_proj(x)            # B T D

        x = self.encoder_proj(x).transpose(1, 2)   # B D T
        # BigVGAN
        # Pre-conv
        x = self.conv_pre(x)
        x = x + self.cond_layer(speaker_embedding)

        if self.encoder2 is None:
            mel_feat_out = self.mel_proj(x.transpose(1,2))  # B D T

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)

            if self.cond_in_each_up_layer:
                x = x + self.conds[i](speaker_embedding)

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x.squeeze(1), (mel_feat_out, None)

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


if __name__ == '__main__':
    """直接使用s3tokenizer得到的codec作为输入，以说话人向量作为condition，重建音频
    """
    import librosa
    import soundfile as sf
    import s3tokenizer
    import torchaudio
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.speaker.speaker_encoder import SpeakerEmbedding

    config_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/conf/cosyvoice_bigvgan_tts.yaml"
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={"use_cuda_kernel": True})
    bigvgan = configs['bigvgan'].cuda()

    ckpt_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/exp/bigvgan_tts/epoch_36_step_580000.pt"
    state_dict = {k.replace('generator.', ''): v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
    bigvgan.load_state_dict(state_dict, strict=False)

    wav_path = "/data/megastore/SHARE/TTS/ref_audios/caikangyong_10s.wav"
    name = os.path.basename(wav_path).split('.')[0]
    wave = torch.from_numpy(
        librosa.load(wav_path, sr=24000)[0]).unsqueeze(0).cuda()  # B T

    speech_tokenzier = s3tokenizer.load_model(
            "speech_tokenizer_v2_25hz", "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/s3tokenizer/").cuda()
    def wav2token(speech_tokenzier, waves_padded, sr_in=24000, wave_lengths=None):
        '''
            waves_padded: B T
        '''
        if not wave_lengths:
            wave_lengths = torch.LongTensor(1).to(waves_padded.device)
            wave_lengths[0] = waves_padded.size(-1)
        mels = []
        batch_size = waves_padded.size(0)
        for i in range(batch_size):
            audio = waves_padded[i, :wave_lengths[i]]
            # whisper speech code use 16k sample_rate
            if sr_in != 16000:
                resampler = torchaudio.transforms.Resample(
                    sr_in, 16000).to(audio.device)
                audio = resampler(audio)
            mels.append(s3tokenizer.log_mel_spectrogram(audio))
        mels, mels_lens = s3tokenizer.padding(mels)
        mels = mels.to(waves_padded.device)
        mels_lens = mels_lens.to(waves_padded.device)
        speech_code, speech_code_len = speech_tokenzier.quantize(
            mels, mels_lens)
        return speech_code, speech_code_len

    speaker_encoder = SpeakerEmbedding(ckpt_path="/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/SpeakerEncoder/speaker_encoder_v2.pt").cuda()
    def wav2spkemb(speaker_encoder, waves_padded):
        wave_lengths = torch.LongTensor(1).to(waves_padded.device)
        wave_lengths[0] = waves_padded.size(-1)
        speaker_embedding = speaker_encoder(waves_padded, wave_lengths)
        return speaker_embedding

    with torch.no_grad():
        speech_token,speech_code_len = wav2token(speech_tokenzier, wave)
        speaker_emb = wav2spkemb(speaker_encoder, wave)
        audio, _ = bigvgan.forward({"speech_token":speech_token,
                                 "speech_token_len":speech_code_len,
                                 "embedding":speaker_emb}, device=wave.device)
        print(f"input: {wave.size()} recon: {audio.size()}")
    sf.write(f"test.wav", audio[0].cpu().detach().numpy(), 24000)