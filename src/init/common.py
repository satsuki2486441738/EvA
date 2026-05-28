"""init/common.py — init 脚本共用的小工具。

只放真正在 ≥2 个 init 文件里重复的逻辑：
  - setup_logger:   统一 logging 配置
  - load_safetensors_shards: 从目录加载 model*.safetensors，合成单个 state_dict
  - report_state_dict_load: 打印 missing/unexpected，并校验非 ced_processor 的 missing
  - copy_files / copy_directory: 文件 / 整目录拷贝，缺失给 warning

不抽"加载 base 模型 → 拷权重 → 保存"这条主流程，架构差异较大，模板化反而难读。
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
    """加载 safetensors 权重。

    优先遵循 HuggingFace 的 model.safetensors.index.json，避免目录中残留或混合命名的
    safetensors 分片被误读。没有 index 时才退回到顶层 model*.safetensors。
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
    """统一打印 load_state_dict(strict=False) 的结果，并校验未预期的 missing。"""
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
    """从 src_dir 拷指定文件名到 dst_dir。缺失给 warning。"""
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
    """整目录拷贝，存在则先删。src 不存在给 warning，不抛错。"""
    log = logger or logging.getLogger(__name__)
    if not os.path.exists(src):
        log.warning(f"Source directory not found: {src}")
        return False
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log.info(f"Copied directory: {src} -> {dst}")
    return True
