# kimi_audio_eva/export_utils.py
# Standalone export logic for KimiAudioModel.
# Separated from model.py to keep model.py focused on model definition.
import os
import shutil
from typing import Optional

import torch

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel
from kimia_infer.models.tokenizer.ced_base.modeling_ced import CedForAudioClassification
from .modeling_kimia import MoonshotKimiaForCausalLM

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def export_model(kimiaudio, output_dir: str, src_submodules=None, base_model_path: str = None):
    """
    Export a KimiAudioModel into three sub-packages under output_dir:
      - output_dir/               : LM (MoonshotKimiaForCausalLM) + source files
      - output_dir/whisper-large-v3/ : Whisper model (base decoder + trained encoder)
      - output_dir/ced-base/      : CED model
      - output_dir/               : tokenizer files (copied from base_model_path)

    base_model_path: root of the pre-trained model directory (used to locate
    whisper-large-v3/, ced-base/, and tokenizer files).
    Defaults to config._name_or_path if not provided.
    """
    if base_model_path is None:
        base_model_path = getattr(getattr(kimiaudio, "config", None), "_name_or_path", None)
    if not base_model_path or not os.path.isdir(base_model_path):
        raise ValueError(
            f"Cannot determine base_model_path for export. Got: {base_model_path!r}. "
            "Pass base_model_path explicitly or ensure config._name_or_path points to "
            "a local directory containing whisper-large-v3/ and ced-base/."
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── LM ──────────────────────────────────────────────────────────────────
    print(f"Saving Kimi-Audio LM to {output_dir}")
    audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
    audio_model = audio_model.to(dtype=torch.bfloat16)
    lm_sd = {
        k: v.to(torch.bfloat16)
        for k, v in kimiaudio.state_dict().items()
        if not k.startswith("whisper_model") and not k.startswith("ced_model")
    }
    audio_model.load_state_dict(lm_sd, strict=False)
    audio_model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)

    # Copy source files from this package (absolute paths, CWD-independent)
    shutil.copyfile(
        os.path.join(_THIS_DIR, "configuration_moonshot_kimia.py"),
        os.path.join(output_dir, "configuration_moonshot_kimia.py"),
    )
    shutil.copyfile(
        os.path.join(_THIS_DIR, "modeling_kimia.py"),
        os.path.join(output_dir, "modeling_moonshot_kimia.py"),
    )

    # ── Whisper ──────────────────────────────────────────────────────────────
    src_whisper = getattr(kimiaudio, "whisper_model", None)
    if src_whisper is None and src_submodules is not None:
        src_whisper = getattr(src_submodules, "whisper_model", None)

    whisper_base_path = os.path.join(base_model_path, "whisper-large-v3")
    whisper_model = WhisperModel.from_pretrained(whisper_base_path)
    if src_whisper is not None:
        enc_sd = {
            k.replace("speech_encoder.", "encoder."): v.to(torch.bfloat16)
            for k, v in src_whisper.state_dict().items() if k.startswith("speech_encoder")
        }
        missing, unexpected = whisper_model.load_state_dict(enc_sd, strict=False)
        assert len(unexpected) == 0, f"Whisper export unexpected keys: {unexpected}"
        for k in missing:
            assert k.startswith("decoder"), f"Whisper export missing non-decoder key: {k}"
    else:
        print("[export_model] whisper_model is None; exporting base whisper weights.")
    whisper_model = whisper_model.to(dtype=torch.bfloat16)
    whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"), torch_dtype=torch.bfloat16)

    # ── CED ──────────────────────────────────────────────────────────────────
    src_ced = getattr(kimiaudio, "ced_model", None)
    if src_ced is None and src_submodules is not None:
        src_ced = getattr(src_submodules, "ced_model", None)

    ced_base_path = os.path.join(base_model_path, "ced-base")
    ced_model = CedForAudioClassification.from_pretrained(ced_base_path)
    if src_ced is not None:
        ced_sd = {
            k.replace("audio_encoder.", "encoder."): v.to(torch.float32)
            for k, v in src_ced.state_dict().items() if k.startswith("audio_encoder")
        }
        missing, unexpected = ced_model.load_state_dict(ced_sd, strict=False)
        assert len(unexpected) == 0, f"CED export unexpected keys: {unexpected}"
    else:
        print("[export_model] ced_model is None; exporting base ced weights.")
    ced_model = ced_model.to(dtype=torch.float32)
    ced_output_dir = os.path.join(output_dir, "ced-base")
    try:
        ced_model.save_pretrained(ced_output_dir, dtype=torch.float32)
    except TypeError:
        ced_model.save_pretrained(ced_output_dir)
    for fname in ("preprocessor_config.json",):
        src = os.path.join(ced_base_path, fname)
        dst = os.path.join(ced_output_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"[export_model] Warning: CED processor file not found: {src}")

    # ── Tokenizer files ───────────────────────────────────────────────────────
    for fname in (
        "special_tokens_map.json",
        "tiktoken.model",
        "tokenization_kimia.py",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"[export_model] Warning: tokenizer file not found: {src}")

    print(f"Exported Kimi-Audio LM, Whisper and CED model to {output_dir}")
