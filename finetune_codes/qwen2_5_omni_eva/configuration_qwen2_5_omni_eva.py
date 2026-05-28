# coding=utf-8
"""Configuration for Qwen2.5-Omni-EvA."""

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)


class Qwen2_5OmniEvaConfig(Qwen2_5OmniThinkerConfig):
    """Audio-only Qwen2.5-Omni Thinker config with EvA CED extensions."""

    model_type = "qwen2_5_omni_eva"

    def __init__(
        self,
        *args,
        use_ced_feature: bool = True,
        ced_processor_input_dim: int = 768,
        ced_freq_bands: int = 4,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        enable_talker: bool = False,
        enable_audio_output: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.use_ced_feature = use_ced_feature
        self.ced_processor_input_dim = ced_processor_input_dim
        self.ced_freq_bands = ced_freq_bands
        self.enable_talker = enable_talker
        self.enable_audio_output = enable_audio_output

    def get_text_config(self, *args, **kwargs):
        return self.text_config
