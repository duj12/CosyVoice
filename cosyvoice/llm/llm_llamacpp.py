#!/usr/bin/env python3
"""
llm_llamacpp.py — 用 llama.cpp (GGUF) 运行 Qwen2 backbone 的 LLM 推理类。

设计：
- LlamaCppEncoder: torch 模块外壳，仅提供 Qwen text embedding（embed_tokens）供 text 分支
  使用，实际 backbone 推理由 llama.cpp GGUF 完成。
- Qwen2LM_Phoneme_LlamaCpp: 与 Qwen2LM_Phoneme_Vllm (llm.py) 同接口的新 LLM 类。
  外部模块（text_embedding/text_encoder/src_attention/llm_embedding/speech_embedding/
  spk_embed_affine_layer/llm_decoder + emotion）保留在 torch，backbone 换成 llama.cpp。

用法（yaml 绑定）：
    llm: !new:cosyvoice.llm.llm_llamacpp.Qwen2LM_Phoneme_LlamaCpp
        ...
        llm: !new:cosyvoice.llm.llm_llamacpp.LlamaCppEncoder
            gguf_path: checkpoints/LAM-VC/LLM/gguf/qwen2-0.5b-lamvc-q8_0.gguf
            n_ctx: 2048
            n_contexts: 6
"""
import logging
import os
import threading
from typing import Callable, Generator, List, Optional, Tuple

# tts/ 目录（checkpoints 根）：acoustics/lam_vc/cosyvoice/llm/ 上四级
_TTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _resolve_checkpoint_path(p: str) -> str:
    """相对路径基于 tts/ 目录解析（与 config.py 的 model_config 一致），绝对路径原样。"""
    if os.path.isabs(p):
        return p
    abs_p = os.path.join(_TTS_DIR, p)
    if os.path.exists(abs_p):
        return abs_p
    # 兜底：相对 cwd
    return p

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama_cpp import LlamaModel, LlamaContextPool
from .llm import SpeakerAdapter
from cosyvoice.transformer.decoder_layer import DecoderLayer
from cosyvoice.transformer.attention import MultiHeadedAttention
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.mask import make_pad_mask

logger = logging.getLogger(__name__)


class LlamaCppEncoder(torch.nn.Module):
    """llama.cpp backbone 外壳。

    持有 Qwen text embedding（供 `text = self.llm.model.model.embed_tokens(text)`），
    backbone 推理走 GGUF（懒加载 llama pool）。不实例化 backbone 层。
    """

    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_contexts: int = 6,
                 flash_attn: bool = True, n_gpu_layers: int = -1,
                 llm_input_size: int = 896, llama_lib_dir: Optional[str] = None):
        super().__init__()
        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        self.n_contexts = n_contexts
        self.flash_attn = flash_attn
        self.n_gpu_layers = n_gpu_layers
        self.llm_input_size = llm_input_size
        self.llama_lib_dir = llama_lib_dir

        # Qwen text embedding 表（从 ckpt 加载 llm.model.model.embed_tokens.weight）
        # 注意：这里只建一个 placeholder，load_state_dict 时填充。词表大小 151936。
        self.model = _EmbedTokensShell(llm_input_size)

        self._pool = None
        self._pool_lock = threading.Lock()

    def ensure_loaded(self):
        """懒加载 llama.cpp model + context pool（首次 inference 或 warmup 时）。"""
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    gguf = _resolve_checkpoint_path(self.gguf_path)
                    logger.info(f"[llamacpp] loading GGUF {gguf}")
                    model = LlamaModel(gguf, n_gpu_layers=self.n_gpu_layers,
                                       lib_dir=self.llama_lib_dir)
                    self._pool = LlamaContextPool(model, n_contexts=self.n_contexts,
                                                  n_ctx=self.n_ctx, flash_attn=self.flash_attn)
                    logger.info(f"[llamacpp] pool ready: {self.n_contexts} ctx, n_embd={model.n_embd}")
        return self._pool

    @property
    def pool(self):
        return self.ensure_loaded()

    def warmup(self):
        """VoiceClone_api 调用 self.model.llm.warmup()，委托到外层类的 warmup。"""
        self.ensure_loaded()


class _EmbedTokensShell(torch.nn.Module):
    """兼容 `self.llm.model.model.embed_tokens` 两层路径的外壳。

    LlamaCppEncoder.model = shell；shell.model = _InnerEmbedTokens（真正持有 embed_tokens）。
    这样 ckpt 键 `llm.model.model.embed_tokens.weight` 精确对应，无多余 missing 键。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.model = _InnerEmbedTokens(dim)


class _InnerEmbedTokens(torch.nn.Module):
    """内层：只含 embed_tokens，供 `self.llm.model.model.embed_tokens(text)` 使用。"""

    def __init__(self, dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, dim)


class Qwen2LM_Phoneme_LlamaCpp(torch.nn.Module):
    """llama.cpp 版 Qwen2LM_Phoneme（接口与 Qwen2LM_Phoneme_Vllm 一致）。"""

    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            text_token_dim: int,
            text_tone_size: int,
            text_tone_dim: int,
            text_lang_size: int,
            text_lang_dim: int,
            text_prsd_size: int,
            text_prsd_dim: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 512,
            src_attn_layers: int = 1,
            use_frontend_prsd: bool = False,
            use_pause_label: bool = False,
            llama_cpp_config: dict = None,
            emotion_num: int = 0,
            non_emotional_label: int = -1,
            add_emotion_before_llm: bool = True,
            emotion_fuse_type: str = 'cat',
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.text_encoder_input_size = text_encoder_input_size
        # 1. build phoneme token inputs related modules
        assert (text_token_dim + text_tone_dim + text_lang_dim + text_prsd_dim) == text_encoder_input_size
        self.text_embedding = nn.ModuleList([
            torch.nn.Embedding(text_token_size, text_token_dim),
            torch.nn.Embedding(text_tone_size, text_tone_dim),
            torch.nn.Embedding(text_lang_size, text_lang_dim),
            torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        ])
        self.use_frontend_prsd = use_frontend_prsd
        self.use_pause_label = use_pause_label
        self.emotion_num = emotion_num
        self.non_emotional_label = non_emotional_label
        self.add_emotion_before_llm = add_emotion_before_llm
        self.emotion_fuse_type = emotion_fuse_type
        logger.info(
            f"llm use prosody: {use_frontend_prsd}, use pause label: {use_pause_label}, "
            f"emotion_num: {emotion_num}, emotion_fuse_type: {emotion_fuse_type}")

        if self.emotion_num > 0:
            self.emotion_embedding = torch.nn.Embedding(self.emotion_num, text_encoder_input_size)
            self.spk_adapter = SpeakerAdapter(dim=llm_input_size, bottleneck=256)
            num_emotions = max(1, self.emotion_num)
            self.emo_adversary = nn.Sequential(
                nn.Linear(llm_input_size, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_emotions)
            )
            if self.add_emotion_before_llm:
                self.emotion_affine_layer = nn.Linear(text_encoder_input_size, llm_input_size, bias=False)

        self.llama_cpp_config = llama_cpp_config or {}
        logger.info(f"llama_cpp_config: {self.llama_cpp_config}")

        self.text_encoder = text_encoder
        if text_encoder is not None:
            self.text_encoder_affine_layer = nn.Linear(
                text_encoder.output_size(), llm_input_size
            )
        else:
            self.text_encoder_affine_layer = nn.Linear(
                text_encoder_input_size, llm_input_size
            )
        #  Hard code Decoder layer as arc-attention
        self.src_attn_layers = src_attn_layers
        self.src_attention = torch.nn.ModuleList([
            DecoderLayer(
                llm_input_size,
                MultiHeadedAttention(16, llm_input_size, 0.1, key_bias=True),
                MultiHeadedAttention(16, llm_input_size, 0.1, key_bias=True),
                PositionwiseFeedForward(llm_input_size, 4096, 0.1),
                dropout_rate=0.1,
                normalize_before=True,
            ) for _ in range(src_attn_layers)
        ])

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm  # LlamaCppEncoder
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    # ------------------------------------------------------------------
    # 权重加载：跳过 backbone（GGUF），只加载外部模块 + Qwen text embedding
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        # 分离 backbone（llm.model.*）与外部模块
        external = {}
        qwen_text_emb_key = None
        qwen_text_emb = None
        for k, v in state_dict.items():
            if k == "llm.model.model.embed_tokens.weight" or k == "llm.model.embed_tokens.weight":
                qwen_text_emb_key = k
                qwen_text_emb = v
            elif k.startswith("llm.model."):
                # backbone 权重（layers/norm/lm_head）——GGUF 已含，跳过
                continue
            else:
                external[k] = v

        # 填 Qwen text embedding（内层 .model.embed_tokens，对应 ckpt 的 llm.model.model.embed_tokens.weight）
        if qwen_text_emb is not None:
            emb = nn.Embedding.from_pretrained(qwen_text_emb, freeze=True)
            self.llm.model.model.embed_tokens = emb
            # 已填充的键也放入 external（值为填充后权重），保证 strict=True 匹配成功
            external[qwen_text_emb_key] = self.llm.model.model.embed_tokens.weight

        # 加载外部模块
        result = super().load_state_dict(external, strict=strict, assign=assign)
        # 校验必需键
        for k in ["text_embedding", "llm_decoder", "speech_embedding", "llm_embedding",
                  "text_encoder_affine_layer", "spk_embed_affine_layer", "src_attention"]:
            missing = [m for m in result.missing_keys if m.startswith(k)]
            if missing and strict:
                logger.warning(f"[llamacpp] missing {k}: {missing[:3]}")
        return result

    # ------------------------------------------------------------------
    # 与 Qwen2LM_Phoneme_Vllm 相同的公共方法
    # ------------------------------------------------------------------
    def encode(self, text, text_lengths):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths,
                                                      decoding_chunk_size=-1,
                                                      num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                logger.warning(f'sampling reaches max_trials {max_trials}')
                break
        return top_ids

    # ------------------------------------------------------------------
    # 推理（async generator，接口与 Qwen2LM_Phoneme_Vllm.inference 一致）
    # ------------------------------------------------------------------
    @torch.inference_mode()
    async def inference(
            self,
            text: Tuple,
            text_len: Tuple,
            prompt_text: Tuple,
            prompt_text_len: Tuple,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 20,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = "",
            emotion_lab: list = [-1, ],
            loracfg=None,
    ) -> Generator[torch.Tensor, None, None]:
        device = embedding.device
        emotion_lab_tensor = torch.tensor(emotion_lab, dtype=torch.long, device=device)

        text, pho = text
        text_len, pho_len = text_len
        prompt_text, prompt_pho = prompt_text
        prompt_text_len, prompt_pho_len = prompt_text_len

        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        pho = torch.concat([prompt_pho, pho], dim=1)
        pho_len += prompt_pho_len

        # pho embedding
        pho_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](pho[:, :, i])
            if not self.use_frontend_prsd and i == 3:
                embed *= 0.0
            pho_embed_list.append(embed)
        pho = torch.cat(pho_embed_list, dim=-1)

        # emotion embedding
        if self.emotion_num > 0:
            emotion_emb_list = []
            for idx, lab in enumerate(emotion_lab):
                if self.non_emotional_label == 0 and lab == -1:
                    lab = 0
                    emotion_lab_tensor[idx] = 0
                if lab < 0:
                    emotion_emb_list.append(
                        torch.zeros(self.text_encoder_input_size).reshape(1, 1, -1).to(device))
                else:
                    emotion_emb_list.append(
                        self.emotion_embedding(torch.LongTensor([lab]).to(device)).reshape(1, 1, -1))
            emotion_emb = torch.cat(emotion_emb_list, dim=0)  # B 1 D
            if self.emotion_fuse_type == 'add':
                pho += emotion_emb

        # 1. encode text
        if self.text_encoder is not None:
            pho, pho_len = self.encode(pho, pho_len)
        else:
            pho = self.text_encoder_affine_layer(pho)

        # Qwen text embedding（保留在 torch，与 llm.py 一致）
        text = self.llm.model.model.embed_tokens(text)

        text_mask = ~make_pad_mask(text_len, text.size(1)).unsqueeze(1).to(device)
        pho_mask = ~make_pad_mask(pho_len, pho.size(1)).unsqueeze(1).to(device)
        for src_attention in self.src_attention:
            pho, pho_mask, text, text_mask = src_attention(pho, pho_mask, text, text_mask)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            labeled_mask = (emotion_lab_tensor >= 0)
            if self.emotion_num > 0 and labeled_mask.any():
                s_hat = embedding.clone()
                s_hat[labeled_mask] = self.spk_adapter(s_hat[labeled_mask])
                embedding = s_hat.unsqueeze(1)
            else:
                embedding = embedding.unsqueeze(1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token.clone())
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat(
            [sos_eos_emb, embedding, pho, task_id_emb, prompt_speech_token_emb],
            dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. llama.cpp step by step decode
        pool = self.llm.pool
        ctx = pool.acquire()
        try:
            ctx.reset()
            # prefill：lm_input [1, T, D] → float32 [T, D]
            # lm_input 是 bf16（autocast），转 float32 传给 llama.cpp（内部 fp32）
            prefill_emb = lm_input.detach().squeeze(0).float().cpu().numpy()
            ctx.prefill(prefill_emb)

            out_tokens = []
            # 第一轮：取 prefill 后的 last hidden；后续：喂上一 speech token embedding
            # （与 llm.py 的 forward_one_step 逐 token 解码语义一致）
            for i in range(max_len):
                if i == 0:
                    y_pred = ctx.get_last_hidden()  # prefill 输出
                else:
                    next_emb = self.speech_embedding.weight[out_tokens[-1]].detach().float().cpu().numpy()
                    y_pred = ctx.decode_step(next_emb)

                # y_pred: [896] float32 → torch → llm_decoder → logits → RAS 采样
                y_t = torch.from_numpy(y_pred).to(device)
                logits = self.llm_decoder(y_t.unsqueeze(0)).float()  # [1, speech_token_size+3]
                top_ids = self.sampling_ids(logits.squeeze(0), out_tokens, sampling,
                                            ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    logger.warning(f"================big token！！！{top_ids}")
                    continue
                # yield token
                yield top_ids
                out_tokens.append(top_ids)
        finally:
            pool.release(ctx)

    # ------------------------------------------------------------------
    # warmup：初始化 GGUF + 池
    # ------------------------------------------------------------------
    def warmup(self):
        self.llm.ensure_loaded()
        pool = self.llm.pool
        # 预热：acquire/release 一个 ctx，做一次 dummy prefill
        ctx = pool.acquire()
        try:
            ctx.reset()
            import numpy as np
            dummy = np.random.randn(16, self.llm_input_size).astype(np.float32)
            ctx.prefill(dummy)
        finally:
            pool.release(ctx)
        logger.info("[llamacpp] warmup done")
