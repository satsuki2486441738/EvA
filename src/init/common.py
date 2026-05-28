"""Shared utilities for init scripts.

Only keep logic that is actually shared by two or more init files:
  - setup_logger: unified logging setup
  - load_safetensors_shards: load model*.safetensors from a directory into one state_dict
  - report_state_dict_load: print missing/unexpected keys and validate missing keys
  - copy_files / copy_directory: copy files or directories with warnings for missing sources

Do not abstract the main "load base model -> copy weights -> save" workflow;
the supported backbones differ enough that a template would be harder to read.
"""

import json
import logging
import os
import shutil
from glob import glob


def setup_logger(name=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)


def load_safetensors_shards(model_path, logger=None):
    """Load safetensors weights.

    Prefer Hugging Face's model.safetensors.index.json so stale or mixed-name
    safetensors shards in the directory are not read accidentally. Fall back to
    top-level model*.safetensors only when the index is absent.
    """
    from safetensors.torch import load_file

    log = logger or logging.getLogger(__name__)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({os.path.join(model_path, name) for name in weight_map.values()})
        log.info(f"Found index with {len(weight_map)} tensors and {len(shards)} shards in {model_path}")
    else:
        abs_root = os.path.abspath(model_path)
        all_shards = sorted(glob(os.path.join(model_path, "model*.safetensors")))
        shards = [s for s in all_shards if os.path.dirname(os.path.abspath(s)) == abs_root]
        log.info(f"Found {len(shards)} safetensors shards in {model_path}")

    sd = {}
    for shard in shards:
        log.info(f"  Loading: {os.path.basename(shard)}")
        shard_sd = load_file(shard, device="cpu")
        if weight_map is None:
            sd.update(shard_sd)
        else:
            wanted = [k for k, v in weight_map.items() if os.path.join(model_path, v) == shard]
            for key in wanted:
                sd[key] = shard_sd[key]
    log.info(f"Total keys: {len(sd)}")
    return sd


def report_state_dict_load(missing, unexpected, expected_missing_substr="ced_processor",
                           logger=None, max_print=10, allowed_missing_substrs=None,
                           fail_on_unexpected_missing=False):
    """Print load_state_dict(strict=False) results and validate unexpected missing keys."""
    log = logger or logging.getLogger(__name__)
    allowed_missing_substrs = allowed_missing_substrs or [expected_missing_substr]
    log.info(f"Missing keys ({len(missing)}):")
    for k in missing[:max_print]:
        log.info(f"  {k}")
    if len(missing) > max_print:
        log.info(f"  ... and {len(missing) - max_print} more")

    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:max_print]:
            log.warning(f"  {k}")
        if len(unexpected) > max_print:
            log.warning(f"  ... and {len(unexpected) - max_print} more")
    else:
        log.info("Unexpected keys: none")

    non_expected = [
        k for k in missing
        if not any(substr in k for substr in allowed_missing_substrs)
    ]
    if non_expected:
        log.warning(f"Unexpected missing keys ({len(non_expected)}):")
        for k in non_expected[:max_print]:
            log.warning(f"  {k}")
        if fail_on_unexpected_missing:
            raise RuntimeError(
                "Unexpected missing keys while loading pretrained weights: "
                + ", ".join(non_expected[:max_print])
            )
    else:
        log.info(f"OK: all missing keys are allowed: {allowed_missing_substrs}")


def copy_files(src_dir, dst_dir, filenames, logger=None):
    """Copy selected files from src_dir to dst_dir, warning when files are missing."""
    log = logger or logging.getLogger(__name__)
    for fname in filenames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log.info(f"  Copied: {fname}")
        else:
            log.warning(f"  Not found (skipped): {fname}")


def copy_directory(src, dst, logger=None):
    """Copy a directory after removing dst if present; warn and return False if src is missing."""
    log = logger or logging.getLogger(__name__)
    if not os.path.exists(src):
        log.warning(f"Source directory not found: {src}")
        return False
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log.info(f"Copied directory: {src} -> {dst}")
    return True
