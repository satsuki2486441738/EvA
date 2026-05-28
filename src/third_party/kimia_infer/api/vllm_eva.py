# kimia_infer/api/vllm_eva.py
"""
vLLM integration for EvA (MoonshotKimiaForCausalLM).

Two components:
  EvAEmbedder       – lightweight, loads only embed_tokens + VQAdaptor + CEDProcessor.
                      Pre-computes full input embeddings (dual-stream fusion formula)
                      for the prompt before vLLM inference.

  EvALLMForVLLM     – extends Qwen2ForCausalLM; registered under the architecture name
                      "MoonshotKimiaForCausalLM" so vLLM's PagedAttention runs the
                      transformer backbone.
                      Overrides get_input_embeddings / forward to add kimia_text_blank
                      embedding during the decode phase (dual-stream text convention).

Usage:
    # Phase 1 – pre-compute embeddings (small footprint)
    embedder = EvAEmbedder(model_path)
    embeds_list = [embedder.compute_inputs_embeds(...) for s in samples]
    del embedder; torch.cuda.empty_cache()

    # Phase 2 – vLLM batch generation
    register_eva_model()
    llm = LLM(model=model_path, trust_remote_code=True, ...)
    outputs = llm.generate(
        [EmbedsPrompt(prompt_embeds=e) for e in embeds_list],
        SamplingParams(...)
    )
"""

import glob
import os
import sys
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

# Stable special-token IDs (see kimia_infer/utils/special_tokens.py)
_KIMIA_TEXT_BLANK_ID: int = 151666


# ──────────────────────────────────────────────────────────────────────────────
# CED resampling helper — re-exported from the shared eva_processor package
# ──────────────────────────────────────────────────────────────────────────────
from eva_processor import resample_proc_to_whisper_timeaware as _resample_proc_to_whisper


# ──────────────────────────────────────────────────────────────────────────────
# EvAEmbedder
# ──────────────────────────────────────────────────────────────────────────────
class EvAEmbedder:
    """
    Loads only the small components needed to compute full inputs_embeds:
      - embed_tokens  (vocab × hidden)
      - VQAdaptor     (whisper feature → hidden)
      - CEDProcessor  (CED features   → hidden)

    Frees itself cleanly: `del embedder; torch.cuda.empty_cache()`.
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoConfig
        from kimi_audio_eva.modeling_kimia import VQAdaptor
        from eva_processor import CEDProcessor

        self.device = torch.device(device)
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Build lightweight modules
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id
        )
        self.vq_adaptor = VQAdaptor(self.config)
        self.ced_processor = CEDProcessor(self.config)

        self._load_weights(model_path)

        for m in (self.embed_tokens, self.vq_adaptor, self.ced_processor):
            m.eval().to(self.device, dtype=torch.bfloat16)

    def _load_weights(self, model_path: str):
        from safetensors.torch import load_file

        sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        modules = {
            "model.embed_tokens.": self.embed_tokens,
            "model.vq_adaptor.":   self.vq_adaptor,
            "model.ced_processor.": self.ced_processor,
        }
        params_cache = {prefix: dict(m.named_parameters()) for prefix, m in modules.items()}

        for sf_path in sf_files:
            sd = load_file(sf_path, device="cpu")
            for key, tensor in sd.items():
                for prefix, params in params_cache.items():
                    if key.startswith(prefix):
                        sub_key = key[len(prefix):]
                        if sub_key in params:
                            with torch.no_grad():
                                params[sub_key].copy_(tensor)
                        break

    @torch.inference_mode()
    def compute_inputs_embeds(
        self,
        audio_input_ids: torch.Tensor,      # [S] or [B, S]
        text_input_ids: torch.Tensor,       # [S] or [B, S]
        is_continuous_mask: torch.Tensor,   # [S] or [B, S] bool
        whisper_input_feature: Optional[torch.Tensor] = None,       # [B, T_w, D_w]
        ced_input_feature: Optional[Tuple[torch.Tensor, ...]] = None,  # (f4, f8, f_last)
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Returns inputs_embeds [B, S, D] (or [S, D] if inputs were 1-D).
        Implements the exact dual-stream fusion from MoonshotKimiaModel.forward.
        """
        squeeze = (audio_input_ids.dim() == 1)
        if squeeze:
            audio_input_ids     = audio_input_ids.unsqueeze(0)
            text_input_ids      = text_input_ids.unsqueeze(0)
            is_continuous_mask  = is_continuous_mask.unsqueeze(0)
            if whisper_input_feature is not None and whisper_input_feature.dim() == 2:
                whisper_input_feature = whisper_input_feature.unsqueeze(0)
            if ced_input_feature is not None:
                ced_input_feature = tuple(
                    f.unsqueeze(0) if f.dim() == 2 else f for f in ced_input_feature
                )

        B = audio_input_ids.shape[0]
        audio_input_ids    = audio_input_ids.to(self.device)
        text_input_ids     = text_input_ids.to(self.device)
        is_continuous_mask = is_continuous_mask.to(self.device)

        # ── Step 1: dual-stream base embedding ──────────────────────────────
        glm = (
            self.embed_tokens(audio_input_ids).to(dtype)
            + self.embed_tokens(text_input_ids).to(dtype)
        )

        # ── Step 2: Whisper / VQAdaptor fusion at continuous positions ───────
        if whisper_input_feature is not None and whisper_input_feature.numel() > 0:
            wf = whisper_input_feature.to(self.device, dtype=dtype)
            whisper_emb = self.vq_adaptor(wf)  # [B, T_w, D]
            placeholder = torch.zeros_like(glm)
            for i in range(B):
                L = int(is_continuous_mask[i].sum())
                if L > 0:
                    placeholder[i, is_continuous_mask[i]] = whisper_emb[i, :L].to(dtype)
            sqrt2 = torch.tensor(2.0, dtype=dtype, device=self.device).sqrt()
            fused = (glm + placeholder) * sqrt2
            glm = torch.where(is_continuous_mask.unsqueeze(-1), fused, glm)

        # ── Step 3: CED / CEDProcessor fusion at continuous positions ─────────
        if ced_input_feature is not None and ced_input_feature[0].numel() > 0:
            f4, f8, fl = ced_input_feature
            f4 = f4.to(self.device, dtype=dtype)
            f8 = f8.to(self.device, dtype=dtype)
            fl = fl.to(self.device, dtype=dtype)
            proc = self.ced_processor((f4, f8, fl))  # [B, T_c, D]
            placeholder = torch.zeros_like(glm)
            for i in range(B):
                L = int(is_continuous_mask[i].sum())
                if L > 0:
                    pooled = _resample_proc_to_whisper(proc[i], L, L * 8).to(dtype)
                    placeholder[i, is_continuous_mask[i]] = pooled
            glm = glm + placeholder

        return glm.squeeze(0) if squeeze else glm


# ──────────────────────────────────────────────────────────────────────────────
# EvALLMForVLLM  –  vLLM model
# ──────────────────────────────────────────────────────────────────────────────
def _make_eva_model_cls():
    """Build EvALLMForVLLM lazily so the module can be imported without vLLM."""
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.sequence import IntermediateTensors

    class EvALLMForVLLM(Qwen2ForCausalLM):
        """
        vLLM Qwen2 model adapted for EvA.

        Key differences from plain Qwen2ForCausalLM:
          * load_weights skips MIMO and audio-adapter weights.
          * get_input_embeddings adds kimia_text_blank embedding so that decode
            steps match EvA's dual-stream convention:
              embed(text_token) + embed(kimia_text_blank)
          * forward uses the above for decode; passes pre-computed inputs_embeds
            unchanged for the prefill (EmbedsPrompt) path.
        """

        _SKIP_PREFIXES = (
            "model.mimo_layers.",
            "model.mimo_norm.",
            "mimo_output.",
            "model.vq_adaptor.",
            "model.ced_processor.",
            "whisper_model.",
            "ced_model.",
        )

        def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Called during decode: add kimia_text_blank embedding."""
            embeds = self.model.embed_tokens(input_ids)            # [T, D]
            blank = torch.full(
                (1,), _KIMIA_TEXT_BLANK_ID,
                dtype=input_ids.dtype, device=input_ids.device
            )
            blank_embed = self.model.embed_tokens(blank)           # [1, D]
            return embeds + blank_embed                            # broadcast

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            if intermediate_tensors is not None:
                # Pipeline parallelism: pass through unchanged
                inputs_embeds = None
            elif inputs_embeds is None:
                # Decode phase: compute dual-stream embeddings
                inputs_embeds = self.get_input_embeddings(input_ids)
                input_ids = None
            # Prefill with EmbedsPrompt: inputs_embeds is already set; use as-is.
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
            skip = self._SKIP_PREFIXES

            def _filtered():
                for name, param in weights:
                    if any(name.startswith(p) for p in skip):
                        continue
                    yield name, param

            return super().load_weights(_filtered())

    return EvALLMForVLLM


try:
    EvALLMForVLLM = _make_eva_model_cls()
except Exception:
    EvALLMForVLLM = None


def register_eva_model() -> None:
    """
    Register EvALLMForVLLM under the architecture name in EvA's config.json.
    Must be called before LLM(...) is instantiated.

    Also patches TikTokenTokenizer for vLLM compatibility: vLLM's
    get_cached_tokenizer accesses all_special_tokens_extended which
    TikTokenTokenizer does not implement.
    """
    if EvALLMForVLLM is None:
        raise RuntimeError("vllm not available or EvALLMForVLLM build failed.")
    from vllm import ModelRegistry
    ModelRegistry.register_model("MoonshotKimiaForCausalLM", EvALLMForVLLM)

    # Patch TikTokenTokenizer if already loaded (fork path).
    # For the spawn path, also patch get_cached_tokenizer so the property
    # is added on first access, before the AttributeError propagates.
    try:
        from vllm.transformers_utils import tokenizer as _vllm_tok
        _orig_cached = _vllm_tok.get_cached_tokenizer

        def _patched_get_cached_tokenizer(tokenizer):
            if not hasattr(tokenizer, "all_special_tokens_extended"):
                type(tokenizer).all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens
                )
            return _orig_cached(tokenizer)

        _vllm_tok.get_cached_tokenizer = _patched_get_cached_tokenizer
    except Exception:
        pass

    # Fix vLLM 0.11.0 typo bug: SlowIncrementalDetokenizer sets
    # self.read_offest (typo) instead of self.read_offset in the
    # prompt-embeds branch, causing AttributeError during decode.
    try:
        from vllm.v1.engine.detokenizer import SlowIncrementalDetokenizer
        _orig_init = SlowIncrementalDetokenizer.__init__

        def _patched_slow_init(self, tokenizer, request):
            _orig_init(self, tokenizer, request)
            if not hasattr(self, "read_offset"):
                self.read_offset = 0

        SlowIncrementalDetokenizer.__init__ = _patched_slow_init
    except Exception:
        pass
