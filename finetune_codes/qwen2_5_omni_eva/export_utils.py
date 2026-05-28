# export_utils.py
# Standalone export logic for Qwen2.5-Omni-EvA.
import os
import shutil
import torch


def export_model(model, output_dir, base_model_path=None):
    """
    Export Qwen2.5-Omni-EvA model:
      - output_dir/: model weights (safetensors)
      - output_dir/ced-base/: CED model (copied from base)
      - output_dir/: tokenizer files (copied from base)
    """
    if base_model_path is None:
        base_model_path = getattr(getattr(model, "config", None), "_name_or_path", None)

    os.makedirs(output_dir, exist_ok=True)

    # save model weights (excluding ced_model which is external)
    sd = {
        k: v.to(torch.bfloat16)
        for k, v in model.state_dict().items()
        if not k.startswith("ced_model")
    }
    model.save_pretrained(
        output_dir,
        state_dict=sd,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # copy ced-base
    if base_model_path and os.path.isdir(base_model_path):
        ced_src = os.path.join(base_model_path, "ced-base")
        ced_dst = os.path.join(output_dir, "ced-base")
        if os.path.exists(ced_src) and not os.path.exists(ced_dst):
            shutil.copytree(ced_src, ced_dst)

        # copy tokenizer files
        for fname in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "merges.txt", "vocab.json", "generation_config.json", "preprocessor_config.json",
        ):
            src = os.path.join(base_model_path, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"Exported Qwen2.5-Omni-EvA model to {output_dir}")
