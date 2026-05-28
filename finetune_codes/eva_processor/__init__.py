# coding=utf-8
"""
eva_processor: reusable modules for EvA's two-stream audio extension.

Shared by both supported backbones:
  - kimi-audio (EvA/kimi_audio_eva)
  - qwen2.5-omni (EvA/qwen2_5_omni_eva)
"""

from .audio_aggregator import AudioAggregator, BertLayer
from .ced_processor import CEDProcessor
from .resampling import resample_proc_to_whisper_timeaware

# Ablation switches controlling whether AudioAggregator frequency gating and
# cross-layer fusion participate in training. True does not change forward
# behavior; False freezes the corresponding parameters in training entries.
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
