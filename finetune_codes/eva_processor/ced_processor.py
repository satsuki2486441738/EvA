# coding=utf-8
"""CEDProcessor: 把 CED encoder 的多层特征融合并投影到 LLM hidden size。"""

import torch
from torch import nn

from .audio_aggregator import AudioAggregator


class CEDProcessor(nn.Module):
    """
    CED 特征处理器。

    config 需要提供：
      - ced_processor_input_dim: int (CED 隐层维度，e.g. 768)
      - hidden_size: int (LLM hidden size，e.g. 4096)
      - ced_freq_bands: int (可选，默认 4)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.ced_processor_input_dim

        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            if isinstance(text_config, dict):
                self.output_dim = text_config["hidden_size"]
            else:
                self.output_dim = text_config.hidden_size
        else:
            self.output_dim = config.hidden_size

        self.audio_aggregator = AudioAggregator(
            freq_bands=getattr(self.config, "ced_freq_bands", 4),
            d_model=self.input_dim,
        )
        self.aggregator_proj_norm = nn.LayerNorm(self.input_dim, elementwise_affine=True)
        self.aggregator_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        # FP32 to avoid BF16 precision truncation eating small gradient updates
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, *args, valid_lengths=None, **kwargs):
        """
        兼容两种调用：
        1) 被某些 wrapper 包裹后：forward(x) 且 x 为 (ced_feat_4, ced_feat_8, ced_feat_last)
        2) 原始关键字：forward(ced_feat_4=..., ced_feat_8=..., ced_feat_last=...)

        valid_lengths（可选）：[B] 张量，池化后每个样本的有效时间步数 T_valid，
        用于在 AudioAggregator 内部屏蔽 padding 位置的 attention，避免污染。
        """
        if len(args) == 1 and not kwargs:
            x = args[0]
            if isinstance(x, (tuple, list)) and len(x) == 3:
                ced_feat_4, ced_feat_8, ced_feat_last = x
            elif isinstance(x, dict):
                ced_feat_4 = x.get("ced_feat_4", None)
                ced_feat_8 = x.get("ced_feat_8", None)
                ced_feat_last = x.get("ced_feat_last", None)
            else:
                raise TypeError(
                    "CEDProcessor.forward expects a tuple/list of (ced_feat_4, ced_feat_8, ced_feat_last) "
                    "or a dict with these keys when called positionally."
                )
        elif len(args) == 3 and not kwargs:
            ced_feat_4, ced_feat_8, ced_feat_last = args
        else:
            ced_feat_4    = kwargs.get("ced_feat_4", None)
            ced_feat_8    = kwargs.get("ced_feat_8", None)
            ced_feat_last = kwargs.get("ced_feat_last", None)

        if ced_feat_4 is None or ced_feat_8 is None or ced_feat_last is None:
            raise TypeError("CEDProcessor.forward missing one of ced_feat_4/ced_feat_8/ced_feat_last")

        # Match the processor parameter dtype. Training/exported Kimi models usually run this
        # block in BF16, while lightweight CPU checks may instantiate it in FP32.
        proc_dtype = self.aggregator_proj_norm.weight.dtype
        ced_feat_4    = ced_feat_4.to(dtype=proc_dtype)
        ced_feat_8    = ced_feat_8.to(dtype=proc_dtype)
        ced_feat_last = ced_feat_last.to(dtype=proc_dtype)

        aggregator_output = self.audio_aggregator(
            ced_feat_last, ced_feat_4, ced_feat_8,
            valid_lengths=valid_lengths,
        )
        aggregator_output = self.aggregator_proj_norm(aggregator_output)
        aggregator_output = self.aggregator_projection(aggregator_output)

        # alpha is FP32; cast result back to BF16 to match LLM dtype
        return (self.alpha * aggregator_output.float()).to(aggregator_output.dtype)
