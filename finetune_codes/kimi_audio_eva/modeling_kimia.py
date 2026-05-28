# modeling_kimia.py
# coding=utf-8
# Copyright 2025 The Moonshot AI Team, Qwen Team, and HuggingFace Inc.
# (license headers unchanged)

# 共享的 EvA processor（AudioAggregator/CEDProcessor/时间对齐函数 + 消融开关）
# 已迁出到独立的 eva_processor 包。这里 re-export 以保持向后兼容。
from eva_processor import (
    AudioAggregator,
    BertLayer,
    CEDProcessor,
    resample_proc_to_whisper_timeaware,
    CED_USE_FREQ_GATE,
    CED_USE_CROSS_LAYER_FUSION,
)

from typing import List, Optional, Tuple, Union
import math
import torch
import torch.utils.checkpoint
from torch import nn

import transformers
from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging
from .configuration_moonshot_kimia import KimiAudioConfig
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2PreTrainedModel,
)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")

logger = logging.get_logger(__name__)


def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(query_layer, key_layer, value_layer, padding_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    num_heads = query_layer.shape[2]

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        padding_mask = padding_mask[:, -query_length:]  # 左填充假设
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, padding_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # 首次调用时无条件重建 cos/sin cache。
        # transformers 5.x 的 from_pretrained 在 init_empty_weights() 上下文中运行 __init__，
        # 所有 tensor 操作产生 meta tensor；persistent=False 的 buffer 不在 checkpoint 里，
        # 物化后可能是 NaN、全零或其他无效值（取决于内存内容），必须无条件重建。
        if not getattr(self, '_rope_ready', False):
            inv_freq = 1.0 / (
                self.base ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device) / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(
                seq_len=self.max_seq_len_cached, device=x.device, dtype=x.dtype
            )
            self._rope_ready = True
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MoonshotAttention(nn.Module):
    """Multi-headed attention"""

    def __init__(self, config: KimiAudioConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # not used by FA, kept for API
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(  bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos[position_ids]
        sin = sin[position_ids]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states   = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states   = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Flash-Attn 要求半精度；若被悄悄升到 FP32，则优先还原为 BF16
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "Input hidden states were cast to float32 (likely due to upcasted norms). "
                "Casting back to bfloat16 for Flash Attention."
            )
            query_states = query_states.to(torch.bfloat16)
            key_states   = key_states.to(torch.bfloat16)
            value_states = value_states.to(torch.bfloat16)

        dropout_rate = 0.0  # 若训练建议配合 dropout

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )
        return attn_output


class MoonshotDecoderLayer(nn.Module):
    def __init__(self, config: KimiAudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        logger.warning_once("using normal flash attention")
        self.self_attn = MoonshotAttention(config=config)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class VQAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


    

class MoonshotKimiaModel(Qwen2PreTrainedModel):
    """
    Transformer decoder
    """
    config_class = KimiAudioConfig

    def __init__(self, config: KimiAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MoonshotDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_whisper_feature = config.use_whisper_feature
        self.use_ced_feature = config.use_ced_feature
        if self.use_whisper_feature:
            self.vq_adaptor = VQAdaptor(config)
        if self.use_ced_feature:
            self.ced_processor = CEDProcessor(config)
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        audio_input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        ced_input_feature: Optional[Tuple[torch.Tensor]] = None,
        ced_valid_lengths: Optional[torch.LongTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if audio_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif audio_input_ids is not None:
            batch_size, seq_length = audio_input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = audio_input_ids.device if audio_input_ids is not None else inputs_embeds.device
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids.to(device)
            else:
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                ).unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            # token embedding
            audio_emb = self.embed_tokens(audio_input_ids)
            text_emb = None
            if text_input_ids is not None:
                text_emb = self.embed_tokens(text_input_ids)

            # 在生成续步（没有音频新帧）时，mask 可能为 None；兜底为全 0
            if is_continuous_mask is None:
                if audio_input_ids is not None:
                    is_continuous_mask = torch.zeros_like(audio_input_ids, dtype=torch.bool)
                elif text_input_ids is not None:
                    is_continuous_mask = torch.zeros_like(text_input_ids, dtype=torch.bool)
                else:
                    # 理论到不了这里，仅防御
                    raise RuntimeError("Cannot infer shape to build is_continuous_mask.")
            is_continuous_mask_expanded = is_continuous_mask.unsqueeze(-1)

            # Keep the no-CED path equivalent to Kimi-Audio: scale only the
            # audio/continuous feature stream, then add text embeddings.
            fused_whisper_placeholder = torch.zeros_like(audio_emb)
            fused_ced_placeholder = torch.zeros_like(audio_emb)
            has_whisper = False
            has_ced = False

            # Whisper 特征填充（BF16）
            if self.use_whisper_feature and whisper_input_feature is not None and whisper_input_feature.numel() > 0:
                whisper_emb = self.vq_adaptor(whisper_input_feature)  # [B, Tw, D]
                for i in range(batch_size):
                    continuous_positions = torch.where(is_continuous_mask[i])[0]
                    num_continuous = len(continuous_positions)
                    if num_continuous > 0:
                        actual_len = min(num_continuous, whisper_emb.shape[1])
                        fused_whisper_placeholder[i, continuous_positions[:actual_len], :] = \
                            whisper_emb[i, :actual_len, :].to(fused_whisper_placeholder.dtype)
                has_whisper = True

            # CED 特征填充（外部 CedEncoder 已 FP32，传入前在 model.py 转为 BF16；CEDProcessor 内部已 * alpha）
            if self.use_ced_feature and ced_input_feature is not None and ced_input_feature[0].numel() > 0:
                ced_flat_4, ced_flat_8, ced_flat_last = ced_input_feature
                proc_160 = self.ced_processor(
                    (ced_flat_4, ced_flat_8, ced_flat_last),
                    valid_lengths=ced_valid_lengths,
                )

                for i in range(batch_size):
                    num_audio_tokens = int(is_continuous_mask[i].sum())
                    if num_audio_tokens > 0:
                        pooled = resample_proc_to_whisper_timeaware(
                            x_t=proc_160[i],
                            feat_len=num_audio_tokens,
                            T_mel=num_audio_tokens * 8,
                        )
                        fused_ced_placeholder[i, is_continuous_mask[i], :] = pooled.to(fused_ced_placeholder.dtype)
                has_ced = True

            glm_features = audio_emb
            if has_whisper or has_ced:
                fused_in_audio = (audio_emb + fused_whisper_placeholder + fused_ced_placeholder) * math.sqrt(2.0)
                glm_features = torch.where(is_continuous_mask_expanded, fused_in_audio, glm_features)
            if text_emb is not None:
                glm_features = glm_features + text_emb

            inputs_embeds = glm_features

        hidden_states = inputs_embeds
        padding_mask = attention_mask

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MoonshotKimiaForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
    }
    _keys_to_ignore_on_save = [r"ced_model\..*", r"whisper_model\..*"]
    _keys_to_ignore_on_load_unexpected = [r"ced_model\..*", r"whisper_model\..*"]
    config_class = KimiAudioConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoonshotKimiaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _initialize_newly_added_modules(self):
        if hasattr(self.model, 'ced_processor'):
            self.model.ced_processor.apply(self._init_weights_for_new_modules)
            logger.info(f"initial success")
        else:
            logger.info("No 'ced_processor' module found, skipping custom initialization.")

    def _init_weights_for_new_modules(self, module):
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        if hasattr(module, 'alpha') and isinstance(getattr(module, 'alpha'), nn.Parameter):
            nn.init.constant_(module.alpha.data, 0.1)
            print(f"Successfully initialized alpha parameter to 0.1")
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
            if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if hasattr(module, 'out_proj') and module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        audio_input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        ced_input_feature: Optional[List[torch.Tensor]] = None,
        ced_valid_lengths: Optional[torch.LongTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_input_feature,
            ced_input_feature=ced_input_feature,
            ced_valid_lengths=ced_valid_lengths,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        text_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (text_logits,) + outputs[1:]
        return CausalLMOutputWithPast(
            loss=None,
            logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
