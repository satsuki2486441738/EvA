# coding=utf-8
"""AudioAggregator: 多频带门控池化 + BertLayer 跨层融合。"""

import torch
from torch import nn


class BertLayer(nn.Module):
    """BERT Layer for cross-attention and feed-forward processing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = getattr(config, 'chunk_size_feed_forward', 0)
        self.seq_len_dim = 1

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, 'hidden_dropout_prob', 0.1),
            batch_first=True
        )
        self.crossattention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, 'hidden_dropout_prob', 0.1),
            batch_first=True
        )
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        key_padding_mask=None,
        encoder_key_padding_mask=None,
        output_attentions=False,
    ):
        attn_input = self.norm1(hidden_states)

        self_attn_output, _ = self.attention(
            attn_input, attn_input, attn_input,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = hidden_states + self.dropout(self_attn_output)

        if encoder_hidden_states is not None:
            cross_attn_input = self.norm2(hidden_states)
            cross_attn_output, _ = self.crossattention(
                cross_attn_input, encoder_hidden_states, encoder_hidden_states,
                attn_mask=encoder_attention_mask,
                key_padding_mask=encoder_key_padding_mask,
            )
            hidden_states = hidden_states + self.dropout(cross_attn_output)

        ffn_input = self.norm3(hidden_states)
        ffn_intermediate = self.intermediate(ffn_input)
        ffn_intermediate = self.activation(ffn_intermediate)
        ffn_output = self.output(ffn_intermediate)
        hidden_states = hidden_states + self.dropout(ffn_output)

        return hidden_states


class AudioAggregator(nn.Module):
    """Audio Aggregator using BERT layers for feature fusion."""
    def __init__(self, freq_bands: int = 4, d_model: int = 768):
        super().__init__()
        self.F = freq_bands
        self.D = d_model

        hid = max(32, d_model // 4)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.SiLU(),
            nn.Linear(hid, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        if getattr(self.gate[-1], "bias", None) is not None:
            nn.init.zeros_(self.gate[-1].bias)

        self.aggregator_config = type('Config', (), {
            'hidden_size': d_model,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'layer_norm_eps': 1e-12
        })()

        self.aggregator_layer_1 = BertLayer(self.aggregator_config)
        self.aggregator_layer_2 = BertLayer(self.aggregator_config)

    def _prepool_freq(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, N, D = x_flat.shape
        if N % self.F != 0:
            return x_flat  # 兜底：无法还原频带轴就不做门控
        T = N // self.F
        x = x_flat.contiguous().view(B, self.F, T, D).transpose(1, 2)  # [B, T, F, D]
        score = self.gate(x)                    # [B, T, F, 1]
        alpha = torch.softmax(score, dim=2)     # [B, T, F, 1]
        y = (alpha * x).sum(dim=2)              # [B, T, D]
        return y

    def forward(self, ced_feat_last, ced_feat_4=None, ced_feat_8=None, valid_lengths=None):
        q  = self._prepool_freq(ced_feat_last)
        k8 = self._prepool_freq(ced_feat_8)
        k4 = self._prepool_freq(ced_feat_4)

        # valid_lengths: [B]，池化后每个样本的有效时间步数 T_valid
        # 构造 key_padding_mask [B, T]：True 表示要被 mask 的 padding 位置
        # 防御：clamp(min=1) 避免全 padding（valid=0）导致 attention softmax 全 -inf → NaN
        key_padding_mask = None
        if valid_lengths is not None:
            T = q.shape[1]
            safe_lengths = valid_lengths.to(q.device).clamp(min=1, max=T)
            idx = torch.arange(T, device=q.device).unsqueeze(0)  # [1, T]
            key_padding_mask = idx >= safe_lengths.unsqueeze(1)  # [B, T]

        out = self.aggregator_layer_1(
            q, encoder_hidden_states=k8,
            key_padding_mask=key_padding_mask,
            encoder_key_padding_mask=key_padding_mask,
        )
        out = self.aggregator_layer_2(
            out, encoder_hidden_states=k4,
            key_padding_mask=key_padding_mask,
            encoder_key_padding_mask=key_padding_mask,
        )
        return out
