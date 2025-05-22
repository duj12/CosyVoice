import os
import json
import torch
import transformers
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from packaging import version
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from typing import Callable, List, Generator, Tuple
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.llm.qwen2_5 import Qwen2ForCausalLM
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.transformer.decoder_layer import DecoderLayer
from cosyvoice.transformer.attention import MultiHeadedAttention
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
import logging
logger = logging.getLogger(__name__)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path, max_cache_len):
        super().__init__()
        self.qwenversion = "2.5" if "3" not in pretrain_path else "3"
        model_class = Qwen2ForCausalLM  # if self.qwenversion == "2.5" else Qwen3ForCausalLM

        if os.path.exists(f"{pretrain_path}/model.safetensors"):
            self.model = model_class.from_pretrained(pretrain_path,
                                                     max_cache_len)
        else:
            config_dict = json.load(open(f"{pretrain_path}/config.json"))
            config = Qwen2Config(attn_implementation='sdpa', **config_dict)
            self.model = model_class(config, max_cache_len, )

    def forward_one_step(self, xs, masks, cache=None):
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2EncoderInfer(Qwen2Encoder):
    def __init__(self, pretrain_path, max_cache_len=5000, dtype=torch.bfloat16):
        Qwen2Encoder.__init__(self, pretrain_path, max_cache_len)
        self.dtype = dtype
        self.qwen_token_embed = deepcopy(self.model.model.embed_tokens)  # 这个embed模块还是保持fp32
        self.model.to(self.dtype)

    @torch.inference_mode()
    def prefill(self, xs):
        # xs:(b,t,c), cache:(None)
        y, cache = self.model(inputs_embeds=xs, cache=None)
        return y, cache, torch.LongTensor([xs.shape[1]]).cuda()

    @torch.inference_mode()
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def forward_one_step(self, xs, cache, cache_position):
        if cache == None:
            xs = xs.to(self.dtype)
            # first step
            return self.prefill(xs)
        else:
            return self.decode(xs, cache, cache_position)
            # y, cache = self.model(inputs_embeds=xs, cache=cache,cache_position=cache_position)
            # return y, cache, cache_position+1

    @torch.inference_mode()
    def forward(self, xs, cache, cache_position):
        return self.model(inputs_embeds=xs, cache=cache,
                          cache_position=cache_position)

    @torch.inference_mode()
    def warmup(self, hidden_size=896):
        self.model.warmup()

    def copy_cache(self, source, target):
        # 新版transformers>=4.51.0的逻辑
        # for id in range(len(source.key_cache)):

        # 旧版transformers<4.51.0的逻辑
        for id in range(len(source.key_cache)):
            target.key_cache[id].copy_(source.key_cache[id])
            target.value_cache[id].copy_(source.value_cache[id])
            if version.Version(transformers.__version__) < version.Version(
                    "4.51.0"):
                getattr(target, f"key_cache_{id}").copy_(
                    getattr(source, f"key_cache_{id}"))
                getattr(target, f"value_cache_{id}").copy_(
                    getattr(source, f"value_cache_{id}"))

    @torch.inference_mode()
    def decode(self, xs, cache, cache_pos):
        y, cache = self.model.forward(graph=True, inputs_embeds=xs, cache=cache)
        return y, cache, cache_pos + 1

    def to(self):
        self.model.to(self.dtype)


class Qwen2LM_Phoneme_Infer(torch.nn.Module):
    '''
    use sec-attention to fuse Text Embedding and Phoneme embedding.
    the sequence used to predict is Phoneme
    '''

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
            use_frontend_prsd: bool = False,
            use_pause_label: bool = False,
            qwen_sglang_config: dict = None,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 1. build phoneme token inputs related modules
        assert (
                           text_token_dim + text_tone_dim + text_lang_dim + text_prsd_dim) == text_encoder_input_size
        self.text_embedding = nn.ModuleList([
            torch.nn.Embedding(text_token_size, text_token_dim),
            torch.nn.Embedding(text_tone_size, text_tone_dim),
            torch.nn.Embedding(text_lang_size, text_lang_dim),
            torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        ])
        self.use_frontend_prsd = use_frontend_prsd
        self.use_pause_label = use_pause_label
        logger.info(
            f"llm use frontend prosody: {use_frontend_prsd}, use pause label: {use_pause_label}")

        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size
        )
        #  Hard code Decoder layer as arc-attention
        self.src_attention = torch.nn.ModuleList([
            DecoderLayer(
                llm_input_size,
                MultiHeadedAttention(16, llm_input_size, 0.1, key_bias=True),
                MultiHeadedAttention(16, llm_input_size, 0.1, key_bias=True),
                PositionwiseFeedForward(llm_input_size, 4096, 0.1),
                dropout_rate=0.1,
                normalize_before=True,
            ) for _ in range(1)
        ])

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)


        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3,
                                                   llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim,
                                                      llm_input_size)

        # 4. sampling method
        self.sampling = sampling

        # sglang 推理时，去掉模型中qwen部分的显存, 将qwen_token_embed剥离出来
        self.qwen_token_embed = self.llm.qwen_token_embed

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths,
                                                      decoding_chunk_size=-1,
                                                      num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token,
                           text_token_len,
                           task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(),
                                    batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(),
                                      batch_first=True)
        lm_input = [torch.concat(
            [sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i],
             task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
            for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input],
                                    dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True,
                                padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                logger.warning(f'sampling reaches max_trials {max_trials} and still get eos when ignore_eos is True, check your input!')
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: Tuple,
            text_len: Tuple,
            prompt_text: Tuple,
            prompt_text_len: Tuple,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = embedding.device

        text, pho = text
        text_len, pho_len = text_len
        prompt_text, prompt_pho = prompt_text
        prompt_text_len, prompt_pho_len = prompt_text_len

        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        pho = torch.concat([prompt_pho, pho], dim=1)
        pho_len += prompt_pho_len

        pho_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](pho[:, :, i])
            if not self.use_frontend_prsd and i == 3:
                embed *= 0.0
            pho_embed_list.append(embed)
        pho = torch.cat(pho_embed_list, dim=-1)

        # 1. encode text
        pho, pho_len = self.encode(pho, pho_len)
        text = self.qwen_token_embed(text)

        text_mask = ~make_pad_mask(text_len, text.size(1)).unsqueeze(
            1)  # (B, 1, T1)
        pho_mask = ~make_pad_mask(pho_len, pho.size(1)).unsqueeze(
            1)  # (B, 1, T2)
        for src_attention in self.src_attention:
            pho, pho_mask, text, text_mask = src_attention(pho, pho_mask, text,
                                                           text_mask)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size,
                                    dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size,
                                                  dtype=text.dtype).to(device)
        lm_input = torch.concat(
            [sos_eos_emb, embedding, pho, task_id_emb, prompt_speech_token_emb],
            dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache, cache_pos = None, None
        for i in range(max_len):
            y_pred, cache, cache_pos = self.llm.forward_one_step(
                lm_input, cache=cache, cache_position=cache_pos)
            # logp = self.llm_decoder(y_pred[:, -1].to(torch.float32)).log_softmax(dim=-1)
            # top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens,
            #                             sampling,
            #                             ignore_eos=True if i < min_len else False).item()
            logits = self.llm_decoder(y_pred[:, -1].to(torch.float32))
            top_ids = self.sampling(logits, out_tokens)

            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                logger.warning(f"================big token！！！{top_ids}")
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
