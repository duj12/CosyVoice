# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
import time
from transformers.models.qwen2 import Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, \
    Qwen2RMSNorm, Qwen2RotaryEmbedding, FlashAttentionKwargs, \
    apply_rotary_pos_emb, Callable, eager_attention_forward, \
    ALL_ATTENTION_FUNCTIONS, Qwen2MLP, repeat_kv
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, \
    BaseModelOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import DynamicCache, StaticCache, \
    SlidingWindowCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from torch.nn.attention import SDPBackend, sdpa_kernel


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_attention_heads * self.head_dim,
                                bias=True)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=True)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,
                                config.hidden_size, bias=False)

        # CUDA GRAPHS
        self.in_graph = torch.cuda.CUDAGraph()
        self.static_in_in = torch.randn(1, 1, config.hidden_size,
                                        dtype=torch.bfloat16, device='cuda')
        self.static_inq_out = torch.randn(1, self.config.num_attention_heads, 1,
                                          self.head_dim, dtype=torch.bfloat16,
                                          device='cuda')
        self.static_ink_out = torch.randn(1, self.config.num_key_value_heads, 1,
                                          self.head_dim, dtype=torch.bfloat16,
                                          device='cuda')
        self.static_inv_out = torch.randn(1, config.num_key_value_heads, 1,
                                          self.head_dim, dtype=torch.bfloat16,
                                          device='cuda')

        self.static_emb1_in = torch.randn(1, 1, 64, dtype=torch.bfloat16,
                                          device='cuda')
        self.static_emb2_in = torch.randn(1, 1, 64, dtype=torch.bfloat16,
                                          device='cuda')

        self.out_graph = torch.cuda.CUDAGraph()
        self.static_out_in = torch.randn(1, 1, config.hidden_size,
                                         dtype=torch.bfloat16, device='cuda')
        self.static_out_out = torch.randn(1, 1, config.hidden_size,
                                          dtype=torch.bfloat16, device='cuda')

    def warmup(self, ):
        with torch.cuda.graph(self.in_graph):
            query_states = self.q_proj(self.static_in_in).view(1, 1,
                                                               self.config.num_attention_heads,
                                                               self.head_dim).transpose(
                1, 2)
            key_states = self.k_proj(self.static_in_in).view(1, 1,
                                                             self.config.num_key_value_heads,
                                                             self.head_dim).transpose(
                1, 2)
            value_states = self.v_proj(self.static_in_in).view(1, 1,
                                                               self.config.num_key_value_heads,
                                                               self.head_dim).transpose(
                1, 2)
            cos, sin = self.static_emb1_in, self.static_emb2_in
            query_states, key_states = apply_rotary_pos_emb(query_states,
                                                            key_states, cos,
                                                            sin)

            self.static_inq_out.copy_(query_states)
            self.static_ink_out.copy_(key_states)
            self.static_inv_out.copy_(value_states)

        with torch.cuda.graph(self.out_graph):
            attn_output = self.o_proj(self.static_out_in)
            self.static_out_out.copy_(attn_output)
        self.attention_interface = torch.compile(
            ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation])

    def forward_with_graph(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # input_shape = hidden_states.shape[:-1]
        # hidden_shape = (*input_shape, -1, self.head_dim)

        input_shape = hidden_states.shape[:-1]
        self.static_in_in.copy_(hidden_states)
        self.static_emb1_in.copy_(position_embeddings[0])
        self.static_emb2_in.copy_(position_embeddings[1])

        self.in_graph.replay()
        cos, sin = position_embeddings
        query_states = self.static_inq_out
        key_states = self.static_ink_out
        value_states = self.static_inv_out

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states,
                                                             value_states,
                                                             self.layer_idx,
                                                             cache_kwargs)

        sliding_window = None

        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]): #不输入mask会自动选择flash attn
        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        self.static_out_in.copy_(attn_output)
        self.out_graph.replay()
        attn_output = self.static_out_out
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(
            1, 2)  # [1,14,1,164]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1,
                                                                             2)  # [1,2,1,64]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(
            1, 2)  # [1,2,1,64]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states,
                                                             value_states,
                                                             self.layer_idx,
                                                             cache_kwargs)

        sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        ################CUDA GRAPHS#########################

        self.graph_mlp = torch.cuda.CUDAGraph()
        self.graph_ln_in = torch.cuda.CUDAGraph()
        self.static_mlp_in = torch.randn(1, 1, self.hidden_size,
                                         dtype=torch.bfloat16, device='cuda')
        self.static_mlp_out = torch.randn(1, 1, self.hidden_size,
                                          dtype=torch.bfloat16, device='cuda')
        self.static_inln_in = torch.randn(1, 1, self.hidden_size,
                                          dtype=torch.bfloat16, device='cuda')
        self.static_inln_out = torch.randn(1, 1, self.hidden_size,
                                           dtype=torch.bfloat16, device='cuda')

    @torch.no_grad()
    def warmup(self, ):
        with torch.cuda.graph(self.graph_ln_in):
            y = self.input_layernorm(self.static_inln_in)
            self.static_inln_out.copy_(y)
        # 强制初始化 CUDA 和 cuBLAS
        y = self.mlp(self.static_mlp_in)
        with torch.cuda.graph(self.graph_mlp):
            y1 = self.post_attention_layernorm(self.static_mlp_in)
            y = self.mlp(y1)
            self.static_mlp_out.copy_(y)
        self.self_attn.warmup()

    def forward_with_graph(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[
                Tuple[torch.Tensor, torch.Tensor]] = None,
            # necessary, but kept here for BC
            **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        self.static_inln_in.copy_(hidden_states)
        self.graph_ln_in.replay()
        hidden_states = self.static_inln_out
        hidden_states, self_attn_weights = self.self_attn.forward_with_graph(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        self.static_mlp_in.copy_(hidden_states)
        self.graph_mlp.replay()
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        hidden_states = self.static_mlp_out
        hidden_states = hidden_states + residual
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[
                Tuple[torch.Tensor, torch.Tensor]] = None,
            # necessary, but kept here for BC
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[
        Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, max_cache_len):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in
             range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.max_cache_len = max_cache_len
        # Initialize weights and apply final processing
        self.post_init()

    def warmup(self, ):
        for layer in self.layers:
            layer.warmup()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            graph=False,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Any] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Any,
    ):
        #########################preforward#######################################
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and past_key_values is None:
            # past_key_values = StaticCache(self.config,max_cache_len=self.max_cache_len,max_batch_size=1,device=inputs_embeds.device,dtype=inputs_embeds.dtype)
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values,
            output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        #####################################################decoder layers################################################
        # decoder layers
        all_hidden_states = None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            foward_func = decoder_layer.forward if not graph else decoder_layer.forward_with_graph
            layer_outputs = foward_func(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output.to_tuple()

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Any,
            output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != \
                                   input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values,
                                                SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
                self.config._attn_implementation == "sdpa"
                and not (using_static_cache or using_sliding_window_cache)
                and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    sliding_window=self.config.sliding_window,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.max_cache_len
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ["cuda", "xpu"]
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask,
                                                                    min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            device: torch.device,
            cache_position: torch.Tensor,
            batch_size: int,
            config: Qwen2Config,
            past_key_values: Any,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype,
                dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length,
                                                device=device) > cache_position.reshape(
                -1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values,
                                  SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length,
                                                       device=device) <= (
                                                  cache_position.reshape(-1,
                                                                         1) - config.sliding_window
                                          )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1,
                                                               -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :,
                               :mask_length] + attention_mask[:, None, None,
                                               :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :,
                                                     :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, max_cache_len):
        super().__init__(config)
        self.model = Qwen2Model(config, max_cache_len)

        # Initialize weights and apply final processing
        self.post_init()

    def warmup(self):
        self.model.warmup()

    def forward(
            self,
            graph=False,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache: Optional[List[torch.FloatTensor]] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            graph=graph,
            attention_mask=None,
            position_ids=None,
            past_key_values=cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )

        return outputs[0], outputs[1]
