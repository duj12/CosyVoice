# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               Jing Du (thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.mask import make_pad_mask, add_optional_chunk_mask
from cosyvoice.transformer.decoder_layer import DecoderLayer
from cosyvoice.transformer.attention import MultiHeadedAttention
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
import logging, random

logger = logging.getLogger(__name__)


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        # while True:
        #     top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        #     if (not ignore_eos) or (self.speech_token_size not in top_ids):
        #         break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()

        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class TransformerLM_MultiCode(torch.nn.Module):
    """
    predict multi-code, maybe with delay-pattern
    """
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        # while True:
        #     top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        #     if (not ignore_eos) or (self.speech_token_size not in top_ids):
        #         break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class TransformerLM_Phoneme(torch.nn.Module):
    """
    input with Phoneme, Tones, Languages, Prosodys
    """
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
            spk_embed_dim: int = 192,
            use_frontend_prsd: bool = True,
            use_pause_label: bool=True,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        assert(text_token_dim+text_tone_dim+text_lang_dim+text_prsd_dim)==text_encoder_input_size
        # self.text_embedding = torch.nn.Embedding(text_token_size, text_token_dim)
        # self.tone_embedding = torch.nn.Embedding(text_tone_size, text_tone_dim)
        # self.lang_embedding = torch.nn.Embedding(text_lang_size, text_lang_dim)
        # self.prsd_embedding = torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        self.text_embedding = nn.ModuleList([
            torch.nn.Embedding(text_token_size, text_token_dim),
            torch.nn.Embedding(text_tone_size, text_tone_dim),
            torch.nn.Embedding(text_lang_size, text_lang_dim),
            torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        ])
        self.use_frontend_prsd = use_frontend_prsd
        self.use_pause_label = use_pause_label
        logger.info(f"llm use frontend prosody: {use_frontend_prsd}, use pause label: {use_pause_label}")

        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        # text_tone = batch['text_tone'].to(device)
        # text_lang = batch['text_lang'].to(device)
        # text_prsd = batch['text_prsd'].to(device)

        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        # pho_embed = self.text_embedding(text_token)
        # tone_embed = self.tone_embedding(text_tone)
        # lang_embed = self.lang_embedding(text_lang)
        # prsd_embed = self.prsd_embedding(text_prsd)
        # text_token = torch.cat([pho_embed,tone_embed,lang_embed,prsd_embed], dim=-1)
        text_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](text_token[:, :, i])
            if not self.use_frontend_prsd and i==3:
                embed *= 0.0
            text_embed_list.append(embed)
        text_token = torch.cat(text_embed_list, dim=-1)

        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):

        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        # while True:
        #     top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        #     if (not ignore_eos) or (self.speech_token_size not in top_ids):
        #         break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        # text = self.text_embedding(text)
        text_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](text[:, :, i])
            if not self.use_frontend_prsd and i==3:
                embed *= 0.0
            text_embed_list.append(embed)
        text = torch.cat(text_embed_list, dim=-1)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class TransformerLM_Phoneme_SpkAdapt(torch.nn.Module):
    """
    input with Phoneme, Tones, Languages, Prosodys
    """
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
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        assert(text_token_dim+text_tone_dim+text_lang_dim+text_prsd_dim)==text_encoder_input_size
        # self.text_embedding = torch.nn.Embedding(text_token_size, text_token_dim)
        # self.tone_embedding = torch.nn.Embedding(text_tone_size, text_tone_dim)
        # self.lang_embedding = torch.nn.Embedding(text_lang_size, text_lang_dim)
        # self.prsd_embedding = torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        self.text_embedding = nn.ModuleList([
            torch.nn.Embedding(text_token_size, text_token_dim),
            torch.nn.Embedding(text_tone_size, text_tone_dim),
            torch.nn.Embedding(text_lang_size, text_lang_dim),
            torch.nn.Embedding(text_prsd_size, text_prsd_dim)
        ])

        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            spk_emb: torch.Tensor,
    ):
        max_length = text.size(1)
        batch_size = text.size(0)
        emb_size = spk_emb.size(1)
        zero_padding = torch.zeros(emb_size).to(spk_emb.device)
        spk_emb_pad = torch.zeros(batch_size, max_length, emb_size).to(spk_emb.device)

        # 遍历每个样本的 text_length 和 spk_emb，填充spk_emb_pad
        for i in range(batch_size):
            length = text_lengths[i]
            emb = spk_emb[i]

            if length < max_length:
                padded_emb = torch.cat(
                    (torch.stack([emb] * length),
                    torch.stack([zero_padding] * (max_length - length))))
            else:
                padded_emb = torch.stack([emb] * length)

            spk_emb_pad[i, :max_length, :] = padded_emb

        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, spk_emb_pad, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        # text_tone = batch['text_tone'].to(device)
        # text_lang = batch['text_lang'].to(device)
        # text_prsd = batch['text_prsd'].to(device)

        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        # pho_embed = self.text_embedding(text_token)
        # tone_embed = self.tone_embedding(text_tone)
        # lang_embed = self.lang_embedding(text_lang)
        # prsd_embed = self.prsd_embedding(text_prsd)
        # text_token = torch.cat([pho_embed,tone_embed,lang_embed,prsd_embed], dim=-1)
        text_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](text_token[:, :, i])
            text_embed_list.append(embed)
        text_token = torch.cat(text_embed_list, dim=-1)

        text_token, text_token_len = self.encode(text_token, text_token_len, embedding)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):

        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        # while True:
        #     top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        #     if (not ignore_eos) or (self.speech_token_size not in top_ids):
        #         break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        # text = self.text_embedding(text)
        text_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](text[:, :, i])
            text_embed_list.append(embed)
        text = torch.cat(text_embed_list, dim=-1)

        # 1. encode text
        text, text_len = self.encode(text, text_len, embedding)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

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
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2LM_Phoneme(torch.nn.Module):
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
            text_emb_mask_prob: float = 0.5,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
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
        self.text_emb_mask_prob = text_emb_mask_prob
        logger.info(f"llm use frontend prosody: {use_frontend_prsd}, use pause label: {use_pause_label}, text_emb_mask_prob:{text_emb_mask_prob}")

        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token,
                           text_token_len, pho_token, pho_token_len,
                           task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(),
                                    batch_first=True)
        pho_token = unpad_sequence(pho_token, pho_token_len.cpu(),
                                    batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(),
                                      batch_first=True)
        lm_input = [torch.concat(
            [sos_eos_emb.squeeze(dim=0), embedding[i],
             text_token[i], pho_token[i],
             task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input],
                                    dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True,
                                padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        pho_token = batch['pho_token'].to(device)
        pho_token_len = batch['pho_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 0. prepare llm_target
        lm_target = [torch.tensor(
            [IGNORE_ID] * (2 + text_token_len[i] + pho_token_len[i]) +
              speech_token[i,:speech_token_len[i]].tolist() +
                [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True,
                                 padding_value=IGNORE_ID).to(device)

        # 1. encode phoneme and text
        pho_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](pho_token[:, :, i])
            if not self.use_frontend_prsd and i == 3:
                embed *= 0.0
            pho_embed_list.append(embed)
        pho_token = torch.cat(pho_embed_list, dim=-1)
        pho_token, pho_token_len = self.encode(pho_token, pho_token_len)

        text_token = self.llm.model.model.embed_tokens(text_token)
        if self.training:
            if random.random() < self.text_emb_mask_prob :
                text_token *= 0.0  # mask text embedding, only depend on phoneme embedding.
            else:  # mask some position of text embedding.
                text_token = torch.nn.Dropout(p=0.2)(text_token)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb, embedding, text_token, text_token_len,
            pho_token, pho_token_len, task_id_emb,
            speech_token, speech_token_len)

        # 6. run lm forward
        # llm_input_mask = torch.tril(torch.ones((
        #     lm_input.size(0), lm_input.size(1), lm_input.size(1)),
        #     device=lm_input.device)).to(torch.bool)   # B T T 三角矩阵，只attention前文. 推理的时候用
        llm_input_mask = ~make_pad_mask(lm_input_len, lm_input.size(1)).unsqueeze(1)  # (B, 1, T)
        # 目前训练时使用全局attention, 不设置动态chunk和固定chunk
        llm_input_mask = add_optional_chunk_mask(
            lm_input, llm_input_mask, use_dynamic_chunk=False,
            use_dynamic_left_chunk=False, decoding_chunk_size=-1,
            static_chunk_size=-1, num_decoding_left_chunks=-1)    # B, T, T

        lm_output, lm_output_mask = self.llm.forward_one_step(lm_input, llm_input_mask.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3),
                          lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}


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
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
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
        text = self.llm.model.model.embed_tokens(text)

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
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, pho, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                # masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                masks=torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device).to(torch.bool),
                cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2LM_Phoneme_Src(torch.nn.Module):
    '''
    use sec-attention to fuse Text Embedding and Phoneme embedding.
    The sequence used to predict is Text BPE
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
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
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
        logger.info(f"llm use frontend prosody: {use_frontend_prsd}, use pause label: {use_pause_label}")

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
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len,
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

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        pho_token = batch['pho_token'].to(device)
        pho_token_len = batch['pho_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 0. prepare llm_target
        lm_target = [torch.tensor(
            [IGNORE_ID] * (2 + text_token_len[i]) +
              speech_token[i,:speech_token_len[i]].tolist() +
                [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True,
                                 padding_value=IGNORE_ID).to(device)

        # 1. encode phoneme and text
        pho_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](pho_token[:, :, i])
            if not self.use_frontend_prsd and i == 3:
                embed *= 0.0
            pho_embed_list.append(embed)
        pho_token = torch.cat(pho_embed_list, dim=-1)
        pho_token, pho_token_len = self.encode(pho_token, pho_token_len)

        text_token = self.llm.model.model.embed_tokens(text_token)
        text_mask = ~make_pad_mask(text_token_len, text_token.size(1)).unsqueeze(1)  # (B, 1, T1)
        pho_mask = ~make_pad_mask(pho_token_len, pho_token.size(1)).unsqueeze(1)  # (B, 1, T2)
        for src_attention in self.src_attention:
            text_token, text_mask, pho_token, pho_mask = src_attention(
                text_token, text_mask, pho_token, pho_mask)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb, embedding, text_token, text_token_len,
            task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        # llm_input_mask = torch.tril(torch.ones((
        #     lm_input.size(0), lm_input.size(1), lm_input.size(1)),
        #     device=lm_input.device)).to(torch.bool)   # B T T 三角矩阵，只attention前文. 推理的时候用
        llm_input_mask = ~make_pad_mask(lm_input_len, lm_input.size(1)).unsqueeze(1)  # (B, 1, T)
        # 目前训练时使用全局attention, 不设置动态chunk和固定chunk
        llm_input_mask = add_optional_chunk_mask(
            lm_input, llm_input_mask, use_dynamic_chunk=False,
            use_dynamic_left_chunk=False, decoding_chunk_size=-1,
            static_chunk_size=-1, num_decoding_left_chunks=-1)    # B, T, T

        lm_output, lm_output_mask = self.llm.forward_one_step(lm_input, llm_input_mask.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3),
                          lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}


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
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
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
        text = self.llm.model.model.embed_tokens(text)

        text_mask = ~make_pad_mask(text_len, text.size(1)).unsqueeze(1)  # (B, 1, T1)
        pho_mask = ~make_pad_mask(pho_len, pho.size(1)).unsqueeze(1)  # (B, 1, T2)
        for src_attention in self.src_attention:
            text, text_mask, pho, pho_mask = src_attention(text, text_mask, pho, pho_mask)


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
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                # masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                masks=torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                 device=lm_input.device).to(torch.bool),
                cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2LM_Phoneme_Src2(torch.nn.Module):
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
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
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
        logger.info(f"llm use frontend prosody: {use_frontend_prsd}, use pause label: {use_pause_label}")

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
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len,
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

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        pho_token = batch['pho_token'].to(device)
        pho_token_len = batch['pho_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 0. prepare llm_target
        lm_target = [torch.tensor(
            [IGNORE_ID] * (2 + pho_token_len[i]) +
              speech_token[i,:speech_token_len[i]].tolist() +
                [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True,
                                 padding_value=IGNORE_ID).to(device)

        # 1. encode phoneme and text
        pho_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](pho_token[:, :, i])
            if not self.use_frontend_prsd and i == 3:
                embed *= 0.0
            pho_embed_list.append(embed)
        pho_token = torch.cat(pho_embed_list, dim=-1)
        pho_token, pho_token_len = self.encode(pho_token, pho_token_len)

        text_token = self.llm.model.model.embed_tokens(text_token)
        text_mask = ~make_pad_mask(text_token_len, text_token.size(1)).unsqueeze(1)  # (B, 1, T1)
        pho_mask = ~make_pad_mask(pho_token_len, pho_token.size(1)).unsqueeze(1)  # (B, 1, T2)
        for src_attention in self.src_attention:
            pho_token, pho_mask, text_token, text_mask = src_attention(
                pho_token, pho_mask, text_token, text_mask)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb, embedding, pho_token, pho_token_len,
            task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        # llm_input_mask = torch.tril(torch.ones((
        #     lm_input.size(0), lm_input.size(1), lm_input.size(1)),
        #     device=lm_input.device)).to(torch.bool)   # B T T 三角矩阵，只attention前文. 推理的时候用
        llm_input_mask = ~make_pad_mask(lm_input_len, lm_input.size(1)).unsqueeze(1)  # (B, 1, T)
        # 目前训练时使用全局attention, 不设置动态chunk和固定chunk
        llm_input_mask = add_optional_chunk_mask(
            lm_input, llm_input_mask, use_dynamic_chunk=False,
            use_dynamic_left_chunk=False, decoding_chunk_size=-1,
            static_chunk_size=-1, num_decoding_left_chunks=-1)    # B, T, T

        lm_output, lm_output_mask = self.llm.forward_one_step(lm_input, llm_input_mask.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3),
                          lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}


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
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
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
        text = self.llm.model.model.embed_tokens(text)

        text_mask = ~make_pad_mask(text_len, text.size(1)).unsqueeze(1)  # (B, 1, T1)
        pho_mask = ~make_pad_mask(pho_len, pho.size(1)).unsqueeze(1)  # (B, 1, T2)
        for src_attention in self.src_attention:
            pho, pho_mask, text, text_mask = src_attention(pho, pho_mask, text, text_mask)

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
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, pho, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                # masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                masks=torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                 device=lm_input.device).to(torch.bool),
                cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)