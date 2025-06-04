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
from cosyvoice.speaker.modules import CausalConv1d
from cosyvoice.utils.mask import add_optional_chunk_mask
import logging

logger = logging.getLogger(__name__)

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
                 up_enc2=None,
                 upsample_first=True,
                 use_dynamic_chunk=False):
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
        self.upsample_first = upsample_first   # True表示encode在upsample之后
        self.use_dynamic_chunk = use_dynamic_chunk
        logger.info(f"VitsDecoder - TextEncoder - upsample_first: {upsample_first}, use_dynamic_chunk: {use_dynamic_chunk}")

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if self.upsample_first:
            if self.up_enc1 is not None:
                x, _ = self.up_enc1(x, x_lengths)   # upsample * 2
                x_lengths = x_lengths * 2
            if self.up_enc2 is not None:
                x, _ = self.up_enc2(x, x_lengths)   # upsample * 2
                x_lengths = x_lengths * 2
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)),
                                 1).to(x.dtype)
        attn_mask = None
        if self.use_dynamic_chunk and self.training:
            attn_mask = add_optional_chunk_mask(
                x.transpose(1, 2), x_mask.bool(), self.use_dynamic_chunk,
                use_dynamic_left_chunk=False, decoding_chunk_size=0,
                static_chunk_size=0, num_decoding_left_chunks=-1,
                max_dynamic_chunk_size=100, enable_full_context=False,
            ).unsqueeze(1)  # 升采样后帧率100  B 1 4T 4T

        x = self.encoder(x * x_mask, x_mask, attn_mask=attn_mask)
        if not self.upsample_first:
            if self.up_enc1 is not None:
                x = x.transpose(1, 2)  # [b, t, h]
                x, _ = self.up_enc1(x, x_lengths)   # upsample * 2
                x_lengths = x_lengths * 2
                x_mask = torch.unsqueeze(commons.sequence_mask(
                    x_lengths, x.size(2)), 1).to(x.dtype)
            if self.up_enc2 is not None:
                x, _ = self.up_enc2(x, x_lengths)   # upsample * 2
                x_lengths = x_lengths * 2
                x = x.transpose(1, 2)  # [b, h, t]
                x_mask = torch.unsqueeze(commons.sequence_mask(
                    x_lengths, x.size(2)), 1).to(x.dtype)

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
                 gin_channels=0,
                 causal=False):
        super().__init__()
        logger.info(f"VitsDecoder Flow is Causal: {causal}")
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
                n_layers, gin_channels=gin_channels, mean_only=True, causal=causal))
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
                 gin_channels=0, causal=False):
        super(Generator, self).__init__()
        logger.info(f"VitsDecoder Generator is Causal: {causal}")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1,
            padding=3) if not causal else CausalConv1d(
            initial_channel, upsample_initial_channel, 7, 1)
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
                self.resblocks.append(resblock(ch, k, d, causal))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False) if not causal else CausalConv1d(
            ch, 1, 7, 1, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if not causal else CausalConv1d(
                gin_channels, upsample_initial_channel, 1)

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
                 token_upsample_ratio=4,
                 up_enc1=None,
                 up_enc2=None,
                 upsample_first=True,
                 use_dynamic_chunk=False,  # True时transformer中添加chunk-aware attn mask
                 causal=False,  # True为因果卷积
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
        self.token_upsample_ratio = token_upsample_ratio   # upsample of codec
        self.hop_length = self.sample_rate // self.frame_rate // self.token_upsample_ratio

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers,
                                 kernel_size, p_dropout,
                                 up_enc1=up_enc1, up_enc2=up_enc2,
                                 upsample_first=upsample_first,
                                 use_dynamic_chunk=use_dynamic_chunk)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                             resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels, causal=causal)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels,
                                      hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)

        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels,
                                          5, 1, 4, gin_channels=gin_channels,
                                          causal=causal)


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

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        min_length = torch.min(y_lengths).item()
        if self.segment_size//self.hop_length > min_length:
            self.segment_size = min_length * self.hop_length
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size//self.hop_length)
        o = self.dec(z_slice, g=g)
        return o, (ids_slice, x_mask, y_mask, z, z_p, m_p, logs_p, m_q, logs_q)

    def inference(self, x, x_lengths, g, noise_scale=0.5, max_len=None):
        g = g.unsqueeze(-1)    # B D 1
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        return o



if __name__ == '__main__':
    """直接使用s3tokenizer得到的codec作为输入，以说话人向量作为condition，重建音频
    """
    import os
    import librosa
    import soundfile as sf
    import s3tokenizer
    import torchaudio
    import librosa, numpy
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.speaker.speaker_encoder import SpeakerEmbedding

    config_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/conf/cosyvoice_vits_tts.yaml"
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={})
    vitsdecoder = configs['vitsdecoder'].cuda().eval()

    ckpt_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/exp/vits_tts/epoch_57_step_1060000.pt"
    state_dict = {k.replace('generator.', ''): v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
    vitsdecoder.load_state_dict(state_dict, strict=False)
    vitsdecoder.dec.remove_weight_norm()

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


    def phase_align(a, b, overlap=1200, sr=24000):
        min_size = len(b)
        n_fft = min(1024, int(min_size//2))
        hop_length = n_fft // 4
        overlap = min(min_size, overlap)
        # 提取重叠部分
        a_tail = a[-overlap:]
        b_head = b[:overlap]

        # 分析a_tail的相位
        stft_a = librosa.stft(a_tail, n_fft=n_fft, hop_length=hop_length)
        mag_a, phase_a = librosa.magphase(stft_a)

        # 分析b_head的幅度和相位
        stft_b = librosa.stft(b_head, n_fft=n_fft, hop_length=hop_length)
        mag_b, phase_b = librosa.magphase(stft_b)

        # 相位对齐：将b_head的相位替换为a_tail的相位
        stft_b_aligned = mag_b * numpy.exp(1j * phase_a)

        # 重建对齐后的b_head
        b_head_aligned = librosa.istft(
            stft_b_aligned, hop_length=hop_length, length=len(b_head))
        b[:overlap] = b_head_aligned
        return b

    def causal_stream_inference(vitsdecoder, speech_token, speaker_emb,
                                base_chunk_size=50, lookahead_size=10,
                                hop_length=960):
        """
        因果形式的伪流式推理
        :param base_chunk_size: 每个chunk输出的基本token数
        :param lookahead_size: 前瞻token数（用于提供未来上下文）
        :param hop_length: 每个token对应的音频采样点数
        """
        B, T = speech_token.shape
        audio_segments = []
        processed_tokens = 0  # 已处理的token位置

        # 第一个chunk特殊处理
        if T > 0:
            # 第一个输入: [0, base_chunk_size + lookahead_size]
            first_end = min(base_chunk_size + lookahead_size, T)
            first_input = speech_token[:, :first_end]
            first_len = torch.tensor([first_end], device=speech_token.device)
            first_audio = vitsdecoder.inference(first_input, first_len,
                                                speaker_emb).squeeze(1)

            # 第一个输出: [0, base_chunk_size] 对应的音频
            first_output_end = base_chunk_size * hop_length
            if first_audio.size(1) < first_output_end:
                # 如果生成的音频不足，使用全部
                audio_segments.append(first_audio)
                processed_tokens = first_end
            else:
                audio_segments.append(first_audio[:, :first_output_end])
                processed_tokens = base_chunk_size

        # 后续chunks处理
        while processed_tokens < T:
            # 计算当前输入结束位置
            input_end = min(processed_tokens + base_chunk_size + lookahead_size,
                            T)

            # 创建从0开始的输入
            current_input = speech_token[:, :input_end]
            current_len = torch.tensor([input_end], device=speech_token.device)

            # 推理当前chunk
            chunk_audio = vitsdecoder.inference(current_input, current_len,
                                                speaker_emb).squeeze(1)

            # 计算需要保留的音频部分（新增部分）
            audio_start = processed_tokens * hop_length
            audio_end = min(processed_tokens + base_chunk_size, T) * hop_length

            # 处理边界情况
            if chunk_audio.size(1) < audio_end:
                # 如果生成的音频不足，取能取到的最大部分
                segment = chunk_audio[:, audio_start:] if chunk_audio.size(
                    1) > audio_start else None
                processed_tokens = T  # 标记处理完成
            else:
                segment = chunk_audio[:, audio_start:audio_end]
                processed_tokens += base_chunk_size

            if segment is not None and segment.size(1) > 0:
                audio_segments.append(segment)

        # 合并所有音频段
        final_audio = torch.cat(audio_segments,
                                dim=1) if audio_segments else torch.zeros(
            (B, 0), device=speech_token.device)

        return final_audio

    def stream_inference0(vitsdecoder, speech_token, speaker_emb, chunk_size=70,
                         overlap_size=10, hop_length=960):
        """
        改进的流式推理函数，支持前后overlap
        :param chunk_size: 每个chunk包含的token数（包含前后overlap）
        :param overlap_size: 前后重叠的token数（前后各占一半）
        :param hop_length: 每个token对应的音频采样点数
        """
        # 计算实际每个chunk输出的基础token数
        base_chunk_size = chunk_size - 2 * overlap_size  # 中间非重叠部分的token数

        B, T = speech_token.shape
        chunks = []
        start = -overlap_size  # 从-overlap_size开始，使第一个块能包含前上下文

        # 创建分块列表
        while start < T:
            end = start + chunk_size
            # 调整起始位置为0如果小于0
            adj_start = max(start, 0)
            # 调整结束位置不超过总长度
            adj_end = min(end, T)

            chunks.append((adj_start, adj_end, start, end))
            start += base_chunk_size  # 移动到下一个基础块位置

        audio_segments = []
        for i, (adj_start, adj_end, orig_start, orig_end) in enumerate(chunks):
            # 获取当前chunk的token
            current_chunk = speech_token[:, adj_start:adj_end]
            current_len = adj_end - adj_start

            # 推理当前chunk
            chunk_code_len = torch.tensor([current_len],
                                          device=speech_token.device)
            chunk_audio = vitsdecoder.inference(current_chunk, chunk_code_len,
                                                speaker_emb).squeeze(1)
            audio_len = chunk_audio.size(1)

            # 计算需要保留的音频范围
            if i == 0:  # 第一个chunk
                # 保留从开始到基础块结束的部分
                keep_start = 0
                keep_end = min(base_chunk_size * hop_length, audio_len)
            elif i == len(chunks) - 1:  # 最后一个chunk
                # 保留从overlap开始到结束的部分
                keep_start = overlap_size * hop_length
                keep_end = audio_len
            else:  # 中间chunk
                # 保留中间基础块部分
                keep_start = overlap_size * hop_length
                keep_end = keep_start + base_chunk_size * hop_length
                # 确保不超过实际长度
                if keep_end > audio_len:
                    keep_end = audio_len

            # 截取需要保留的音频段
            segment = chunk_audio[:, keep_start:keep_end]
            # if len(audio_segments) > 0:
            #     last_segment = audio_segments[-1].squeeze().cpu().numpy()
            #     segment = torch.from_numpy(phase_align(last_segment, segment.squeeze().cpu().numpy())).unsqueeze(0).to(speech_token.device) # 对当前chunk进行相位修复
            audio_segments.append(segment)

        # 合并所有音频段
        final_audio = torch.cat(audio_segments,
                                dim=1) if audio_segments else torch.zeros(
            (B, 0), device=speech_token.device)

        return final_audio


    def stream_inference(vitsdecoder, speech_token, speaker_emb, chunk_size=70,
                         overlap_size=10, hop_length=960):
        """
        改进的流式推理函数，支持前后overlap和淡入淡出效果
        :param chunk_size: 每个chunk包含的token数（包含前后overlap）
        :param overlap_size: 前后重叠的token数（前后各占一半）
        :param hop_length: 每个token对应的音频采样点数
        """
        # 计算实际每个chunk输出的基础token数
        base_chunk_size = chunk_size - 2 * overlap_size  # 中间非重叠部分的token数
        fade_length = 2* overlap_size * hop_length  # 淡入淡出窗口长度

        B, T = speech_token.shape
        chunks = []
        start = -overlap_size  # 从-overlap_size开始，使第一个块能包含前上下文

        # 创建分块列表
        while start < T:
            end = start + chunk_size
            # 调整起始位置为0如果小于0
            adj_start = max(start, 0)
            # 调整结束位置不超过总长度
            adj_end = min(end, T)

            chunks.append((adj_start, adj_end, start, end))
            start += base_chunk_size  # 移动到下一个基础块位置

        audio_segments = []
        prev_tail = None  # 保存上一个块的尾部（用于交叉淡化）

        for i, (adj_start, adj_end, orig_start, orig_end) in enumerate(chunks):
            # 获取当前chunk的token
            current_chunk = speech_token[:, adj_start:adj_end]
            current_len = adj_end - adj_start

            # 推理当前chunk
            chunk_code_len = torch.tensor([current_len],
                                          device=speech_token.device)
            chunk_audio = vitsdecoder.inference(current_chunk, chunk_code_len,
                                                speaker_emb).squeeze(1)
            audio_len = chunk_audio.size(1)

            # 计算需要保留的音频范围
            if i == 0:  # 第一个chunk
                # 保留从开始到基础块结束的部分
                keep_start = 0
                keep_end = min(base_chunk_size * hop_length, audio_len)
                segment = chunk_audio[:, keep_start:keep_end]

                # 保存尾部用于下一个块的交叉淡化, 是末尾overlap + 多生成的部分
                tail_start = max(0, audio_len - fade_length)
                prev_tail = chunk_audio[:, tail_start:]

            elif i == len(chunks) - 1:  # 最后一个chunk
                # 保留从overlap开始到结束的部分
                keep_start = overlap_size * hop_length
                keep_end = audio_len

                # 如果有上一个块的尾部，应用交叉淡化
                if prev_tail is not None:
                    # 当前块的头部用于交叉淡化
                    head_end = min(fade_length, audio_len)
                    current_head = chunk_audio[:, :head_end]

                    # 创建淡入淡出曲线
                    fade_out = torch.linspace(1.0, 0.0, prev_tail.size(1),
                                              device=chunk_audio.device)
                    fade_in = torch.linspace(0.0, 1.0, current_head.size(1),
                                             device=chunk_audio.device)

                    # 调整长度以匹配（取两者中较小的长度）
                    min_len = min(prev_tail.size(1), current_head.size(1))
                    mixed = prev_tail[:, :min_len] * fade_out[
                                                     :min_len] + current_head[:,
                                                                 :min_len] * fade_in[
                                                                             :min_len]

                    # 拼接：混合部分 + 当前块的剩余部分
                    segment = torch.cat(
                        [mixed[:, keep_start:], chunk_audio[:, min_len:keep_end]], dim=1)
                else:
                    segment = chunk_audio[:, keep_start:keep_end]
            else:  # 中间chunk
                # 保留中间基础块部分
                keep_start = overlap_size * hop_length
                keep_end = keep_start + base_chunk_size * hop_length
                if keep_end > audio_len:
                    keep_end = audio_len

                # 如果有上一个块的尾部，应用交叉淡化
                if prev_tail is not None:
                    # 当前块的头部用于交叉淡化
                    head_end = min(fade_length, audio_len)
                    current_head = chunk_audio[:, :head_end]

                    # 创建淡入淡出曲线
                    fade_out = torch.linspace(1.0, 0.0, prev_tail.size(1),
                                              device=chunk_audio.device)
                    fade_in = torch.linspace(0.0, 1.0, current_head.size(1),
                                             device=chunk_audio.device)

                    # 调整长度以匹配（取两者中较小的长度）
                    min_len = min(prev_tail.size(1), current_head.size(1))
                    mixed = prev_tail[:, :min_len] * fade_out[
                                                     :min_len] + current_head[:,
                                                                 :min_len] * fade_in[
                                                                             :min_len]

                    # 拼接：混合部分 + 当前块的基础部分
                    segment = torch.cat(
                        [mixed[:, keep_start:], chunk_audio[:, min_len:keep_end]], dim=1)

                else:
                    segment = chunk_audio[:, :keep_end]

                # 保存尾部用于下一个块的交叉淡化
                tail_start = max(0, audio_len - fade_length)
                prev_tail = chunk_audio[:, tail_start:]

            audio_segments.append(segment)

        # 合并所有音频段
        final_audio = torch.cat(audio_segments,
                                dim=1) if audio_segments else torch.zeros(
            (B, 0), device=speech_token.device)

        return final_audio


    def save_chunks_to_wav(chunks, output_path, sample_rate=24000,
                           bit_depth=16, n_channels=1):
        """将多个音频chunk保存为WAV文件"""
        import io
        import struct
        # 1. 合并所有chunk的PCM数据
        pcm_data = b''.join(
            (chunk * (2 ** (bit_depth - 1) - 1)).astype(
                numpy.int16).tobytes()
             for chunk in chunks
        )

        # 2. 计算文件信息
        data_size = len(pcm_data)
        byte_rate = sample_rate * n_channels * bit_depth // 8
        block_align = n_channels * bit_depth // 8

        # 3. 构建WAV文件头 (44字节)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',  # RIFF标识
            36 + data_size,  # 文件总大小-8
            b'WAVE',  # WAVE标识
            b'fmt ',  # fmt块标识
            16,  # fmt块大小
            1,  # 音频格式（PCM=1）
            n_channels,  # 声道数
            sample_rate,  # 采样率
            byte_rate,  # 字节率
            block_align,  # 块对齐
            bit_depth,  # 位深度
            b'data',  # data块标识
            data_size  # 数据大小
        )

        # 4. 写入文件
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(pcm_data)

    with torch.no_grad():
        test_dir = "/data/megastore/SHARE/TTS/ref_audios/codec_test"
        save_dir = "/data/megastore/SHARE/TTS/ref_audios/codec_test_result"
        for file in os.listdir(test_dir):
            name = os.path.splitext(file)[0]
            wav_path = os.path.join(test_dir, file)
            wave = torch.from_numpy(
                librosa.load(wav_path, sr=24000)[0]).unsqueeze(0).cuda()  # B T

            speech_token,speech_code_len = wav2token(speech_tokenzier, wave)
            speaker_emb = wav2spkemb(speaker_encoder, wave)
            audio = vitsdecoder.inference(speech_token, speech_code_len, speaker_emb).squeeze(1)
            print(f"input: {wave.size()} recon: {audio.size()}")
            save_path = f"{save_dir}/{name}_vits.wav"
            sf.write(save_path, audio[0].cpu().detach().numpy(), 24000)
            audio_stream = stream_inference(vitsdecoder,speech_token,speaker_emb, 60, 5)
            save_path = f"{save_dir}/{name}_vits_stream.wav"
            sf.write(save_path, audio_stream[0].cpu().detach().numpy(), 24000)

            def split_audio_array(audio_array, chunk_size=48000):
                """更高效的实现"""
                n = len(audio_array)
                # 创建完整chunk的列表
                chunks = [
                    audio_array[i:i + chunk_size]
                    for i in range(0, n - chunk_size + 1, chunk_size)
                ]

                # 添加最后一个chunk（可能不足长度）
                last_start = n - (n % chunk_size)
                if last_start < n:
                    chunks.append(audio_array[last_start:])

                return chunks
            # chunks = split_audio_array(audio_stream[0].cpu().detach().numpy())
            # save_chunks_to_wav(chunks, save_path)  # 不用sf.write, 直接转二进制写入
            # for i, chunk in enumerate(chunks):
            #     save_path = f"{save_dir}/{name}_vits_stream_chunk_{i+1}.wav"
            #     sf.write(save_path, chunk, 24000)