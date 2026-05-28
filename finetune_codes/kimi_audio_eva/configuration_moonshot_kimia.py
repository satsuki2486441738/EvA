from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class KimiAudioConfig(Qwen2Config):
    def __init__(
        self,
        vocab_size=163840,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        kimia_text_output_vocab: int = 152064,
        num_audio_special_tokens: int = 512,
        num_base_tokens: int = 151643,
        kimia_token_offset: int = 152064,
        use_whisper_feature: bool = True,
        kimia_adaptor_input_dim: int = 5120,
        use_ced_feature: bool = True,
        ced_processor_input_dim: int = 768,
        kimia_media_begin: int = 151661,
        kimia_media_end: int = 151663,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **kwargs,
        )

        self.rope_theta = rope_theta
        # vocab
        self.kimia_text_output_vocab = kimia_text_output_vocab
        self.num_audio_special_tokens = num_audio_special_tokens
        self.num_base_tokens = num_base_tokens
        self.kimia_token_offset = kimia_token_offset
        # features
        self.use_whisper_feature = use_whisper_feature
        self.kimia_adaptor_input_dim = kimia_adaptor_input_dim
        self.use_ced_feature = use_ced_feature
        self.ced_processor_input_dim = ced_processor_input_dim
        # special tokens
        self.kimia_media_begin = kimia_media_begin
        self.kimia_media_end = kimia_media_end