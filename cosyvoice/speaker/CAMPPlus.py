# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
#           2023     Jing Du (thuduj12@163.com)
# All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict
import torch
from torch import nn
import struct
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as Kaldi
from scipy.ndimage.morphology import binary_dilation
import numpy as np


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, \
                'Expect equal paddings, but got even kernel size ({})'.format(
                    kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(
            self, bn_channels, out_channels,  kernel_size,
            stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True)+self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y*m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(
                x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(
                x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, \
            'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels, bn_channels=bn_channels,
                kernel_size=kernel_size, stride=stride,
                dilation=dilation, bias=bias,
                config_str=config_str, memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def trim_long_silences(wav, sampling_rate=16000, vad_window_length=30,
                       vad_moving_average_width=8, vad_max_silence_length=6,
                       ):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away
            (length <= original wav length)
    """
    int16_max = (2 ** 15) - 1
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav),
                           *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    import webrtcvad
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                          sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate(
            (np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask,
                                 np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
        trim_sil: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor
        self.trim_sil = trim_sil


    def __call__(self, wav, sr=16000, dither=0):
        if sr != self.sample_rate:
            import librosa
            wav_resmp = librosa.resample(
                wav.cpu().numpy(), orig_sr=sr, target_sr=self.sample_rate)
            wav_resmp = torch.from_numpy(wav_resmp).to(wav)
            wav = wav_resmp
        if self.trim_sil:
            wave_trim = trim_long_silences(wav.cpu().numpy())
            wave_trim = torch.from_numpy(wave_trim).to(wav)
            wav = wave_trim

        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class FCM(nn.Module):
    def __init__(self,
                block=BasicResBlock,
                num_blocks=[2, 2],
                m_channels=32,
                feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels,
                               kernel_size=3, stride=(2, 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1]*shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(self,
                 feat_dim=80,
                 embedding_size=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True,
                 need_grad=True):
        super(CAMPPlus, self).__init__()
        self.need_grad = need_grad
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                    zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers, in_channels=channels,
                out_channels=growth_rate, bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size, dilation=dilation,
                config_str=config_str, memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: B x D x T, if not BDT, you should transpose it.
        :return: embedding.
        """
        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)

        return x


class CosineClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, config_str='batchnorm')
            )
            input_dim = inter_dim

        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x


class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        self.nonlinear = nn.ReLU(inplace=True)
        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, bias=True)
            )
            input_dim = inter_dim

        self.linear = nn.Linear(input_dim, out_neurons, bias=True)

    def forward(self, x):
        # x: [B, dim]
        x = self.nonlinear(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.linear(x)
        return x


# test CAM++ speaker embedding model
if __name__ == "__main__":
    import json
    import os

    def load_wav(wav_file, obj_fs=16000):
        import torchaudio
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(
                f'[WARNING]: The sample rate of '
                f'{wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav


    def compute_embedding(wav_file, embedding_model,
                          cuda=False, save_dir=None):
        import numpy as np
        # load wav
        wav = load_wav(wav_file)
        feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0)
        feat = feat.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        if cuda:
            embedding_model = embedding_model.to('cuda')
            feat = feat.to('cuda')
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()

        if save_dir is not None:
            save_path = save_dir / (
                    '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
            print(
                f'[INFO]: The extracted embedding from '
                f'{wav_file} is saved to {save_path}.')

        return embedding

    # TO BE SET, the pretrained sv model dir.
    ckpt_dir = 'pretrained_models/speaker_encoder/speech_campplus_sv_zh-cn_16k-common'

    config_file = os.path.join(ckpt_dir, "configuration.json")
    config = json.load(open(config_file, 'r', encoding='utf-8'))
    model_config = config['model']['model_config']
    model_ckpt = config['model']['pretrained_model']
    model_ckpt = os.path.join(ckpt_dir, model_ckpt)
    model_state_dict = torch.load(model_ckpt, map_location='cpu')

    campplus = CAMPPlus(feat_dim=model_config['fbank_dim'],
                        embedding_size=model_config['emb_size'])
    campplus.load_state_dict(model_state_dict)
    campplus.eval()

    input_wav1 = os.path.join(ckpt_dir, 'examples/speaker1_a_cn_16k.wav')
    input_wav2 = os.path.join(ckpt_dir, 'examples/speaker1_b_cn_16k.wav')
    input_wav3 = os.path.join(ckpt_dir, 'examples/speaker2_a_cn_16k.wav')
    import time
    time_s = time.perf_counter()
    for i in range(10):
        embedding1 = compute_embedding(input_wav1, campplus, cuda=True)
        embedding2 = compute_embedding(input_wav2, campplus, cuda=True)
        embedding3 = compute_embedding(input_wav3, campplus, cuda=True)
    time_e = time.perf_counter()
    print(f"total time {time_e - time_s}")
    # compute similarity score
    print('[INFO]: Computing the similarity score...')
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    scores = similarity(torch.from_numpy(embedding1),
                        torch.from_numpy(embedding2)).item()
    print('[INFO]: The similarity score between '
          'two input wavs of same person is %.4f' % scores)

    scores = similarity(torch.from_numpy(embedding1),
                        torch.from_numpy(embedding3)).item()
    print('[INFO]: The similarity score between '
          'two input wavs of different person is %.4f' % scores)