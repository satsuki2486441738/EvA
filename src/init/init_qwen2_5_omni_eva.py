#!/usr/bin/env python3
"""
init_qwen2_5_omni_eva.py

Initialize Qwen2.5-Omni-EvA-base by combining EvA's CED architecture with Qwen2.5-Omni-7B weights.

Usage:
    python src/init/init_qwen2_5_omni_eva.py \
        --qwen_omni_path /path/to/Qwen2.5-Omni-7B \
        --ced_path /path/to/ced-base \
        --output_dir /path/to/Qwen2.5-Omni-EvA-Base
"""

import argparse
import os
import sys

import torch

_INIT_DIR = os.path.dirname(__file__)
if _INIT_DIR not in sys.path:
    sys.path.insert(0, _INIT_DIR)

from common import (
    setup_logger,
    load_safetensors_shards,
    report_state_dict_load,
    copy_files,
    copy_directory,
)

logger = setup_logger(__name__)


def load_qwen_omni_weights(model_path):
    """Load Qwen2.5-Omni weights, keeping only Thinker's audio_tower, model, and lm_head."""
    state_dict = load_safetensors_shards(model_path, logger=logger)

    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("thinker.audio_tower."):
            new_key = key.replace("thinker.", "")
            filtered_state_dict[new_key] = value
        elif key.startswith("thinker.model."):
            new_key = key.replace("thinker.", "")
            filtered_state_dict[new_key] = value
        elif key.startswith("thinker.lm_head."):
            new_key = key.replace("thinker.", "")
            filtered_state_dict[new_key] = value

    logger.info(f"Filtered keys: {len(filtered_state_dict)}")
    return filtered_state_dict


def main():
    parser = argparse.ArgumentParser(description="Initialize Qwen2.5-Omni-EvA-base")
    parser.add_argument("--qwen_omni_path", type=str,
                        required=True,
                        help="Path to Qwen2.5-Omni-7B model directory")
    parser.add_argument("--ced_path", type=str,
                        required=True,
                        help="Path to CED model directory")
    parser.add_argument("--output_dir", type=str,
                        required=True,
                        help="Output directory for the Qwen2.5-Omni-EvA-base model")
    args = parser.parse_args()

    from qwen2_5_omni_eva.configuration_qwen2_5_omni_eva import Qwen2_5OmniEvaConfig
    from qwen2_5_omni_eva.modeling_qwen2_5_omni_eva import Qwen2_5OmniEvaForConditionalGeneration
    from transformers import AutoConfig

    # 2. Load Qwen2.5-Omni config.
    logger.info(f"Loading Qwen2.5-Omni config from: {args.qwen_omni_path}")
    qwen_omni_config = AutoConfig.from_pretrained(args.qwen_omni_path, trust_remote_code=True)

    # 3. Create Qwen2.5-Omni-EvA config.
    logger.info("Creating Qwen2.5-Omni-EvA config...")

    # Extract audio/vision/text configs and token/rope settings from thinker_config.
    thinker_config = qwen_omni_config.thinker_config
    if hasattr(thinker_config, 'audio_config'):
        audio_cfg = thinker_config.audio_config
        vision_cfg = thinker_config.vision_config
        text_cfg = thinker_config.text_config
    else:
        # Dict-style config fallback.
        audio_cfg = thinker_config.get('audio_config') if isinstance(thinker_config, dict) else thinker_config.audio_config
        vision_cfg = thinker_config.get('vision_config') if isinstance(thinker_config, dict) else thinker_config.vision_config
        text_cfg = thinker_config.get('text_config') if isinstance(thinker_config, dict) else thinker_config.text_config

    config = Qwen2_5OmniEvaConfig(
        audio_config=audio_cfg,
        vision_config=vision_cfg,
        text_config=text_cfg,
        audio_token_index=getattr(thinker_config, "audio_token_index", 151646),
        image_token_index=getattr(thinker_config, "image_token_index", 151655),
        video_token_index=getattr(thinker_config, "video_token_index", 151656),
        position_id_per_seconds=getattr(thinker_config, "position_id_per_seconds", 25),
        seconds_per_chunk=getattr(thinker_config, "seconds_per_chunk", 2),
        audio_start_token_id=getattr(thinker_config, "audio_start_token_id", 151647),
        audio_end_token_id=getattr(thinker_config, "audio_end_token_id", 151648),
        vision_start_token_id=getattr(thinker_config, "vision_start_token_id", 151652),
        vision_end_token_id=getattr(thinker_config, "vision_end_token_id", 151653),
        user_token_id=getattr(thinker_config, "user_token_id", 872),
        initializer_range=getattr(thinker_config, "initializer_range", 0.02),
        tie_word_embeddings=getattr(thinker_config, "tie_word_embeddings", False),
        pad_token_id=getattr(thinker_config, "pad_token_id", 151643),
        bos_token_id=getattr(thinker_config, "bos_token_id", 151644),
        eos_token_id=getattr(thinker_config, "eos_token_id", 151645),
        use_ced_feature=True,
        ced_processor_input_dim=768,
        ced_freq_bands=4,
        enable_talker=False,
        enable_audio_output=False,
    )
    logger.info(f"  use_ced_feature={config.use_ced_feature}")
    logger.info(f"  ced_processor_input_dim={config.ced_processor_input_dim}")

    # 4. Instantiate the model with random initialization.
    logger.info("Instantiating Qwen2_5OmniEvaForConditionalGeneration (random init)...")
    model = Qwen2_5OmniEvaForConditionalGeneration(config)

    # 5. Initialize ced_processor weights.
    logger.info("Initializing ced_processor weights...")
    model._initialize_newly_added_modules()

    # 6. Load Qwen2.5-Omni weights.
    logger.info(f"Loading Qwen2.5-Omni weights from: {args.qwen_omni_path}")
    qwen_omni_sd = load_qwen_omni_weights(args.qwen_omni_path)

    # 7. Apply weights with strict=False.
    logger.info("Loading Qwen2.5-Omni weights into EvA model (strict=False)...")
    missing, unexpected = model.load_state_dict(qwen_omni_sd, strict=False)
    report_state_dict_load(
        missing,
        unexpected,
        logger=logger,
        allowed_missing_substrs=["ced_processor"],
        fail_on_unexpected_missing=True,
    )

    # 8. Convert to BF16.
    logger.info("Converting model to bfloat16...")
    model.to(torch.bfloat16)

    # 9. Save model.
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving model weights (bfloat16) to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")

    # 10. Copy CED model.
    copy_directory(
        args.ced_path,
        os.path.join(args.output_dir, "ced-base"),
        logger=logger,
    )

    # 11. Copy tokenizer files.
    logger.info("Copying tokenizer files...")
    copy_files(args.qwen_omni_path, args.output_dir, [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "generation_config.json",
        "preprocessor_config.json",
    ], logger=logger)

    # 12. Save config.
    logger.info("Saving config...")
    config.save_pretrained(args.output_dir)

    logger.info(f"\nDone! Qwen2.5-Omni-EvA-base model saved to: {args.output_dir}")
    logger.info("To verify:")
    logger.info("  python src/inference/infer_qwen2_5_omni_eva.py --model_path <output_dir> --audio_path <wav>")


if __name__ == "__main__":
    main()
