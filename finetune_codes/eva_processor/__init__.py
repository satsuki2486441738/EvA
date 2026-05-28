# coding=utf-8
"""
eva_processor: EvA 双流外挂的可复用模块。

被两套基础架构共享：
  - kimi-audio (EvA/kimi_audio_eva)
  - qwen2.5-omni (EvA/qwen2_5_omni_eva)
"""

from .audio_aggregator import AudioAggregator, BertLayer
from .ced_processor import CEDProcessor
from .resampling import resample_proc_to_whisper_timeaware

# 消融开关：控制 AudioAggregator 中频带门控和跨层融合是否参与训练。
# 注意：True 不改变前向行为，False 会在外部训练入口中冻结对应参数。
CED_USE_FREQ_GATE = True
CED_USE_CROSS_LAYER_FUSION = True

__all__ = [
    "AudioAggregator",
    "BertLayer",
    "CEDProcessor",
    "resample_proc_to_whisper_timeaware",
    "CED_USE_FREQ_GATE",
    "CED_USE_CROSS_LAYER_FUSION",
]
