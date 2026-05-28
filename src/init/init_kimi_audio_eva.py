#!/usr/bin/env python3
"""
init_kimi_audio_eva.py

Initialize EvA-base by combining EvA's architecture with Kimi-Audio-7B weights.

Usage:
    python src/init/init_kimi_audio_eva.py \
        --kimi_path /path/to/Kimi-Audio-7B \
        --eva_config_path /path/to/eva-config \
        --output_dir /path/to/Kimi-Audio-EvA-Base
"""

import argparse
import os
import sys

import torch

from common import (
    setup_logger,
    load_safetensors_shards,
    report_state_dict_load,
    copy_files,
    copy_directory,
)

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Initialize EvA-base from Kimi-Audio-7B weights")
    parser.add_argument("--kimi_path", type=str, required=True,
                        help="Path to Kimi-Audio-7B model directory")
    parser.add_argument("--eva_config_path", type=str, required=True,
                        help="Path to EvA config/model directory (provides config.json and ced-base/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the EvA-base model")
    args = parser.parse_args()

    from kimi_audio_eva.configuration_moonshot_kimia import KimiAudioConfig
    from kimi_audio_eva.modeling_kimia import MoonshotKimiaForCausalLM

    # 2. Load EvA config
    logger.info(f"Loading EvA config from: {args.eva_config_path}")
    config = KimiAudioConfig.from_pretrained(args.eva_config_path)
    logger.info(f"  use_ced_feature={config.use_ced_feature}")
    logger.info(f"  ced_processor_input_dim={config.ced_processor_input_dim}")
    logger.info(f"  kimia_mimo_audiodelaytokens={config.kimia_mimo_audiodelaytokens}")

    # 3. Instantiate model with EvA architecture (random weights)
    logger.info("Instantiating MoonshotKimiaForCausalLM with EvA config (random init)...")
    model = MoonshotKimiaForCausalLM(config)

    # 4. Initialize newly-added modules (ced_processor) with proper weights
    logger.info("Initializing ced_processor weights...")
    model._initialize_newly_added_modules()

    # 5. Load all Kimi-Audio-7B safetensors shards
    logger.info(f"Loading Kimi-Audio-7B weights from: {args.kimi_path}")
    kimi_sd = load_safetensors_shards(args.kimi_path, logger=logger)

    # 6. Apply Kimi weights with strict=False
    #    Matching keys load normally; ced_processor keys have no match → stay as initialized
    logger.info("Loading Kimi weights into EvA model (strict=False)...")
    missing, unexpected = model.load_state_dict(kimi_sd, strict=False)
    report_state_dict_load(
        missing,
        unexpected,
        allowed_missing_substrs=[
            "ced_processor",
            "rotary_emb.inv_freq",
        ],
        fail_on_unexpected_missing=True,
        logger=logger,
    )

    # 7. Convert to bfloat16 before saving (Kimi weights are bfloat16;
    #    the model was instantiated in float32, so we must cast explicitly)
    logger.info("Converting model to bfloat16...")
    model.to(torch.bfloat16)

    # 8. Save LLM weights in 4 shards (matching EvA layout)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving model weights (bfloat16, 4 shards) to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")

    # 9. Copy non-weight files / sub-directories from eva_config_path
    logger.info("Copying files from eva_config_path...")
    copy_files(args.eva_config_path, args.output_dir, [
        "special_tokens_map.json",
        "tiktoken.model",
        "tokenizer_config.json",
        "tokenization_kimia.py",
        "generation_config.json",
        "modeling_moonshot_kimia.py",
        "configuration_moonshot_kimia.py",
    ], logger=logger)

    copy_directory(
        os.path.join(args.eva_config_path, "whisper-large-v3"),
        os.path.join(args.output_dir, "whisper-large-v3"),
        logger=logger,
    )
    copy_directory(
        os.path.join(args.eva_config_path, "ced-base"),
        os.path.join(args.output_dir, "ced-base"),
        logger=logger,
    )

    # 10. Save EvA config — overwrites the config.json produced by save_pretrained
    logger.info("Saving EvA config (overwriting save_pretrained's config.json)...")
    config.save_pretrained(args.output_dir)

    logger.info(f"\nDone! EvA-base model saved to: {args.output_dir}")
    logger.info("To verify:")
    logger.info("  python -c \"")
    logger.info("  from kimi_audio_eva.model import KimiAudioModel")
    logger.info(f"  m = KimiAudioModel.init_from_pretrained('{args.output_dir}', {{}})")
    logger.info("  print('OK, ced_processor loaded:', hasattr(m.model, 'ced_processor'))")
    logger.info("  \"")


if __name__ == "__main__":
    main()
