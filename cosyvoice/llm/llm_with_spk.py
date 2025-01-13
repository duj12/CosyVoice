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
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy

class VICReg(nn.Module):
    """https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py"""
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        """
        :param x: size(B, D)
        :param y: size(B, D)
        :return:
        """

        batch_size, feat_dim = x.size()

        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 +\
                   torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(feat_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(feat_dim)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


class TransformerLM_Phoneme(torch.nn.Module):
    """
    input with Phoneme, Tones, Languages, Prosodys
    Train with the Speaker Encoder Module
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
            speaker_encoder: torch.nn.Module,
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

        self.speaker_embed = speaker_encoder
        if self.speaker_embed.spec_aug_config is not None and self.training:
            self.VICReg_loss = VICReg()

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

    def encode_speaker(self,
                       wave: torch.Tensor,
                       wave_len: torch.Tensor,
    ):
        spk_audio_crop = self.speaker_embed.spk_audio_crop
        if spk_audio_crop:
            wave_len = wave_len.to('cpu')
            crop_length = spk_audio_crop * self.speaker_embed.sampling_rate
            extracted_waves = []
            spk_wave_len = []
            for b, true_length in enumerate(wave_len):
                if true_length < crop_length:  # 需要拼接至crop_length
                    repeat_times = (crop_length + wave_len[b] - 1) // wave_len[b]
                    extracted_wave = torch.cat([wave[b][:true_length]] * repeat_times)
                    extracted_wave = extracted_wave[:crop_length]
                    spk_wave_len.append(crop_length)
                else:
                    random_length = torch.randint(crop_length, true_length+1, (1,)).item()
                    start_idx = torch.randint(0, true_length-random_length+1, (1,)).item()
                    extracted_wave = wave[b, start_idx:start_idx + random_length]
                    spk_wave_len.append(random_length)

                extracted_waves.append(extracted_wave)

            spk_wave = pad_sequence(extracted_waves, batch_first=True, padding_value=0)
            spk_wave = spk_wave.to(wave.device)
            spk_wave_len = torch.tensor(spk_wave_len).to(wave.device)
        else:
            spk_wave = wave
            spk_wave_len = wave_len

        # the speaker_embed_model use 24k wave tensor input, if not 24k, resample is needed
        speaker_embs = self.speaker_embed(spk_wave.unsqueeze(1), spk_wave_len)  # B D
        return speaker_embs

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
        wave = batch['speech'].to(device)
        wave_len = batch['speech_len'].to(device)
        text_token = batch['text_token'].to(device)

        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)

        # 0. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_embed_list = []
        for i in range(len(self.text_embedding)):
            embed = self.text_embedding[i](text_token[:, :, i])
            text_embed_list.append(embed)
        text_token = torch.cat(text_embed_list, dim=-1)

        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding_ori = self.encode_speaker(wave, wave_len)
        batch['embedding'] = embedding_ori
        embedding = F.normalize(embedding_ori, dim=1)
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

        if self.speaker_embed.spec_aug_config is not None and self.training:
            with torch.no_grad():
                embedding_aug = self.encode_speaker(wave, wave_len)
            vic_reg_loss = self.VICReg_loss(embedding_ori, embedding_aug)
            loss += vic_reg_loss

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
