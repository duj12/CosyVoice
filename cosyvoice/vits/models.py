import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

import cosyvoice.vits.attentions as attentions
import cosyvoice.speaker.modules as modules
import cosyvoice.speaker.commons as commons
from cosyvoice.speaker.commons import init_weights


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 up_enc1=None,
                 up_enc2=None):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.up_enc1 = up_enc1
        self.up_enc2 = up_enc2
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x, _ = self.up_enc1(x, x_lengths)   # upsample * 2
        x, _ = self.up_enc2(x, x_lengths*2)   # upsample * 2
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths*4, x.size(2)),
                                 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate,
                n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate,
            n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1,
                               padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class VitsDecoder(nn.Module):
    """
    Vits decoder for codec-to-audio reconstruction
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 segment_size,
                 gin_channels=512,
                 sample_rate=24000,
                 frame_rate=25,
                 up_enc1=None,
                 up_enc2=None,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate    # 25
        self.token_upsample_ratio = 4   # fixed 4x upsample of codec
        self.hop_length = self.sample_rate // self.frame_rate // self.token_upsample_ratio

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers,
                                 kernel_size, p_dropout,
                                 up_enc1=up_enc1, up_enc2=up_enc2)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                             resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels,
                                      hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)

        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels,
                                          5, 1, 4, gin_channels=gin_channels)


    def forward(self, batch, device):
        x = batch['speech_token'].to(device)                 # B T
        x_lengths = batch['speech_token_len'].to(device)
        y = batch['speech_feat'].to(device).transpose(1,2)   # B D T
        y_lengths = batch['speech_feat_len'].to(device)
        g = batch['embedding'].unsqueeze(-1).to(device)      # B D 1

        # fix codec len and feat len mismatch
        max_feat_len = y.size(-1)
        max_token_len = max_feat_len // self.token_upsample_ratio
        token = x[:, :max_token_len]
        token_len = torch.where(x_lengths > max_token_len, torch.tensor(max_token_len), x_lengths)
        max_feat_len = max_token_len * self.token_upsample_ratio
        feat = y[:, :, :max_feat_len]
        feat_len = torch.where(y_lengths > max_feat_len, torch.tensor(max_feat_len), y_lengths)
        x, x_lengths = token, token_len
        y, y_lengths = feat, feat_len

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)  # 4x upsample and encode
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size//self.hop_length)
        o = self.dec(z_slice, g=g)
        return o, (ids_slice, x_mask, y_mask, z, z_p, m_p, logs_p, m_q, logs_q)

    def inference(self, x, x_lengths, g, noise_scale=0.5, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        return o
