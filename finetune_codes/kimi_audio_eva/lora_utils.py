# kimi_audio_eva/lora_utils.py
import os
import gc
import types
import inspect
import logging
from typing import List, Optional

import torch
from torch import nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .model import KimiAudioModel
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# ------------------------
# 小工具
# ------------------------
def rank0_print(*args):
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        print(*args)
        return
    if torch.distributed.get_rank() == 0:
        print(*args)


def print_trainable_params(model):
    total = trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    rank0_print(f"Total params: {total/1e6:.1f} M | Trainable: {trainable/1e6:.3f} M ({100*trainable/total:.2f} %)")
    for n,p in model.named_parameters():
        if p.requires_grad:
            rank0_print(n, p.shape)

def _unwrap_ddp(model):
    return getattr(model, "module", model)


# ------------------------
# LoRA 目标选择
# ------------------------
def collect_lora_targets(
    model: nn.Module,
    lora_modules: Optional[List[str]] = None,
    lora_target_layers: Optional[List[str]] = None,
    excluded_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    - lora_modules: 限制 LoRA 作用范围的模块前缀列表（默认 ["model.layers"]）
    - lora_target_layers: 在 lora_modules 范围内匹配的层类型（默认 attention 投影）
    - excluded_prefixes: 强制排除的前缀（trainable_modules + 音频编码器）
    """
    excluded = list(excluded_prefixes or []) + ["whisper_model", "ced_model"]

    def _excluded(name: str) -> bool:
        return any(name.startswith(p) for p in excluded)

    if not lora_modules:
        lora_modules = ["model.layers"]
    if not lora_target_layers:
        lora_target_layers = ["q_proj", "k_proj", "v_proj", "o_proj"]

    layer_types = set(lora_target_layers)
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _excluded(name):
            continue
        if not any(name.startswith(p) for p in lora_modules):
            continue
        if name.split(".")[-1] not in layer_types:
            continue
        names.append(name)

    return sorted(set(names))


# ------------------------
# PEFT/Transformers 接口适配
# ------------------------
def _ensure_prepare_inputs_for_generation(model: nn.Module) -> nn.Module:
    """
    PEFT 在构建时会读取 pifg，避免它强行要求 input_ids。
    多模态场景直接"原样透传"即可。
    """
    if hasattr(model, "prepare_inputs_for_generation"):
        return model

    def _prepare_inputs_for_generation(self, **kwargs):
        return {k: v for k, v in kwargs.items()}

    model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)
    return model


def _patch_peft_forward_filter_kwargs(peft_model):
    """
    某些 PEFT 包装会吞掉不认识的 kwargs，这里做一层白名单过滤后再转发给 base_model。
    """
    base = peft_model.get_base_model()
    try:
        allowed = set(inspect.signature(base.forward).parameters.keys())
    except Exception:
        allowed = set()

    allowed |= {
        "use_cache", "output_attentions", "output_hidden_states", "return_dict",
        "position_ids", "attention_mask", "generation_mode", "past_key_values", "inputs_embeds",
        "audio_input_ids", "text_input_ids", "waveform", "waveform_lengths", "is_continuous_mask",
        "labels",
    }

    def _forward(self, *args, **kwargs):
        fkwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return self.base_model(*args, **fkwargs)

    peft_model.forward = types.MethodType(_forward, peft_model)
    logger.info(f"[LoRA] Patched PeftModel.forward to filter kwargs. Allowed: {sorted(list(allowed))}")


# ------------------------
# 组装 LoRA
# ------------------------
def attach_lora(model: KimiAudioModel, lora_args) -> nn.Module:
    """
    lora_args 需要具备属性：
      - lora_r, lora_alpha, lora_dropout, adapter_name
      - lora_modules: LoRA 作用范围前缀（逗号分隔，默认 model.layers）
      - lora_target_layers: LoRA 层类型（逗号分隔，默认 q_proj,k_proj,v_proj,o_proj）
      - trainable_modules: 全参可训模块前缀（逗号分隔，默认 None）
    """
    lora_modules = [m.strip() for m in lora_args.lora_modules.split(",") if m.strip()] \
        if getattr(lora_args, "lora_modules", None) else None
    lora_target_layers = [l.strip() for l in lora_args.lora_target_layers.split(",") if l.strip()] \
        if getattr(lora_args, "lora_target_layers", None) else None
    trainable_mods = [m.strip() for m in lora_args.trainable_modules.split(",") if m.strip()] \
        if getattr(lora_args, "trainable_modules", None) else []

    model = _ensure_prepare_inputs_for_generation(model)

    targets = collect_lora_targets(
        model,
        lora_modules=lora_modules,
        lora_target_layers=lora_target_layers,
        excluded_prefixes=trainable_mods,
    )
    if len(targets) == 0:
        raise RuntimeError("No LoRA target modules found.")

    lconf = LoraConfig(
        r=getattr(lora_args, "lora_r", 16),
        lora_alpha=getattr(lora_args, "lora_alpha", 32),
        lora_dropout=getattr(lora_args, "lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    peft_model = get_peft_model(model, lconf, adapter_name=getattr(lora_args, "adapter_name", "default"))
    peft_model._plain_modules_to_save = []

    _patch_peft_forward_filter_kwargs(peft_model)

    # 冻结所有，再打开 LoRA 和 trainable_modules 的参数
    for _, p in peft_model.named_parameters():
        p.requires_grad = False
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    if trainable_mods:
        for n, p in peft_model.named_parameters():
            if "lora_" not in n and any(pref in n for pref in trainable_mods):
                p.requires_grad = True

    return peft_model


# ------------------------
# 合并 LoRA 并导出三包
# ------------------------
def _build_cpu_shadow_model_and_merge(original_model: nn.Module) -> nn.Module:
    """
    在 CPU 上重建同构 base -> 载入当前训练权重 -> 若有 LoRA，merge_and_unload ->
    再把 modules_to_save 覆盖回 merged base。
    """
    base_or_peft = _unwrap_ddp(original_model)

    # 1) 用同构 config 重建 CPU 空壳（whisper/ced 保持 None，由 src_submodules 在 export 时提供，
    #    避免 init_from_pretrained 将它们强制加载到 CUDA 导致显存翻倍）
    base_cfg = base_or_peft.get_base_model().config if isinstance(base_or_peft, PeftModel) else base_or_peft.config
    base_cpu = KimiAudioModel(base_cfg)
    base_cpu = base_cpu.to(dtype=torch.float32)

    base_cpu.eval()
    _ensure_prepare_inputs_for_generation(base_cpu)

    # 2) 抓当前权重到 CPU
    with torch.no_grad():
        sd_cpu = {k: v.detach().to("cpu") for k, v in base_or_peft.state_dict().items()}

    # 3) 非 PEFT：直接加载
    if not isinstance(base_or_peft, PeftModel):
        _ = base_cpu.load_state_dict(sd_cpu, strict=False)
        return base_cpu

    # 4) 拿激活 adapter 的配置
    peft_cfg_map = base_or_peft.peft_config
    active_name = getattr(base_or_peft, "active_adapter", None) or next(iter(peft_cfg_map.keys()))
    active_cfg = peft_cfg_map[active_name]

    # 5) 在 CPU base 注入 LoRA，再加载训练权重
    from peft import get_peft_model as _get_peft_model
    shadow_peft_cpu = _get_peft_model(base_cpu, active_cfg, adapter_name=active_name)
    try:
        shadow_peft_cpu.set_adapter(active_name)
    except Exception:
        pass

    missing, unexpected = shadow_peft_cpu.load_state_dict(sd_cpu, strict=False)
    if len(unexpected) > 0:
        logger.warning(f"[Export] Unexpected keys when loading into CPU PEFT: {len(unexpected)}")
    if len(missing) > 0:
        logger.warning(f"[Export] Missing keys when loading into CPU PEFT: {len(missing)}")

    # 6) 合并 LoRA
    with torch.no_grad():
        merged_cpu = shadow_peft_cpu.merge_and_unload()
    merged_cpu.eval()

    # 7) 拷贝 modules_to_save 的权重回 merged_cpu
    modules_to_copy = set()
    try:
        cfg = peft_cfg_map[active_name]
        if getattr(cfg, "modules_to_save", None):
            modules_to_copy.update(cfg.modules_to_save)
    except Exception:
        pass
    modules_to_copy.update(getattr(base_or_peft, "_plain_modules_to_save", []) or [])

    if modules_to_copy:
        dst = merged_cpu.state_dict()
        moved = 0

        def strip_base(k: str) -> str:
            return k[11:] if k.startswith("base_model.") else k

        for k, v in sd_cpu.items():
            k2 = strip_base(k)
            if any(k2.startswith(pref) for pref in modules_to_copy) and (k2 in dst):
                dst[k2] = v
                moved += 1
        merged_cpu.load_state_dict(dst, strict=False)
        logger.info(f"[Export] Copied {moved} parameters from modules_to_save into merged CPU base.")
    else:
        logger.info("[Export] No modules_to_save to copy into merged CPU base.")

    return merged_cpu


def export_split_from_model(model_or_wrapper: nn.Module, export_dir: str) -> str:
    """
    合并 LoRA -> 调用 KimiAudioModel.export_model 导出三包（LM/whisper/ced）。
    src_submodules 传入**当前训练模型**，以便导出其 whisper/ced 子模块的权重。
    """
    os.makedirs(export_dir, exist_ok=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    merged_cpu = _build_cpu_shadow_model_and_merge(model_or_wrapper)

    KimiAudioModel.export_model(
        merged_cpu,
        export_dir,
        src_submodules=_unwrap_ddp(model_or_wrapper),
    )
    logger.info(f"[Export] Exported split model to: {export_dir}")
    return export_dir


# ------------------------
# 回调：按 epoch 导出
# ------------------------
class ExportSplitCallback(TrainerCallback):
    """
    每个 epoch 结束后导出一次 split（已融合 LoRA，兼容 DDP）。
    只在主进程 (process_index==0) 执行导出，导出前后插 barrier 确保所有进程同步。
    """
    def __init__(
        self,
        export_base_dir: str,
        every_n_epochs: int = 1,
        keep_last_k: Optional[int] = None,
    ):
        super().__init__()
        self.export_base_dir = export_base_dir
        self.every = max(1, int(every_n_epochs))
        self.keep_last_k = keep_last_k
        os.makedirs(self.export_base_dir, exist_ok=True)
        self._epoch_dirs: List[str] = []

    def _barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return
        ep = int(state.epoch)
        if ep % self.every != 0:
            return

        self._barrier()

        if getattr(args, "process_index", 0) == 0:
            model = kwargs.get("model", None)
            if model is None:
                logger.warning("[ExportSplitCallback] 'model' not in kwargs; skipping.")
            else:
                out_dir = os.path.join(self.export_base_dir, f"epoch_{ep:03d}")
                print(f"[ExportSplitCallback] Exporting split model for epoch {ep} -> {out_dir}")

                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

                export_split_from_model(model, out_dir)
                self._epoch_dirs.append(out_dir)
                if self.keep_last_k is not None and len(self._epoch_dirs) > self.keep_last_k:
                    import shutil
                    to_rm = self._epoch_dirs.pop(0)
                    try:
                        shutil.rmtree(to_rm, ignore_errors=True)
                        print(f"[ExportSplitCallback] Removed old split dir: {to_rm}")
                    except Exception as e:
                        logger.warning(f"[ExportSplitCallback] Failed to remove {to_rm}: {e}")

        self._barrier()


# ------------------------
# 回调：按 step 导出（供 GRPO 使用）
# ------------------------
class ExportSplitByStepCallback(TrainerCallback):
    """
    每隔 `every_n_steps` 优化步导出一次 split（已融合 LoRA）。
    - 只在主进程 (process_index==0) 导出
    - 训练结束时兜底再导出一次（如最后一步未正好命中步频）
    """
    def __init__(self, export_base_dir: str, every_n_steps: int = 100, keep_last_k: Optional[int] = None):
        super().__init__()
        assert every_n_steps > 0, "every_n_steps must be > 0"
        self.export_base_dir = export_base_dir
        self.every_n_steps = int(every_n_steps)
        self.keep_last_k = keep_last_k
        os.makedirs(self.export_base_dir, exist_ok=True)
        self._step_dirs: List[str] = []
        self._last_export_step: int = -1

    def _maybe_cleanup(self):
        if self.keep_last_k is not None and len(self._step_dirs) > self.keep_last_k:
            import shutil
            to_rm = self._step_dirs.pop(0)
            try:
                shutil.rmtree(to_rm, ignore_errors=True)
                print(f"[ExportSplitByStepCallback] Removed old split dir: {to_rm}")
            except Exception as e:
                logger.warning(f"[ExportSplitByStepCallback] Failed to remove {to_rm}: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        # 只在完成一个"优化步"（global_step 增加）后被调用
        if getattr(args, "process_index", 0) != 0:
            return
        gs = int(state.global_step or 0)
        if gs <= 0:
            return
        if gs % self.every_n_steps != 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        out_dir = os.path.join(self.export_base_dir, f"step_{gs:06d}")
        print(f"[ExportSplitByStepCallback] Exporting split model for step {gs} -> {out_dir}")

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        export_split_from_model(model, out_dir)
        self._step_dirs.append(out_dir)
        self._last_export_step = gs
        self._maybe_cleanup()

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束兜底导出一次（如果最后一步没有命中步频）
        if getattr(args, "process_index", 0) != 0:
            return
        gs = int(state.global_step or 0)
        if gs <= 0:
            return
        if self._last_export_step == gs:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        out_dir = os.path.join(self.export_base_dir, f"step_{gs:06d}_final")
        print(f"[ExportSplitByStepCallback] Final export split model at step {gs} -> {out_dir}")

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        export_split_from_model(model, out_dir)
        self._step_dirs.append(out_dir)
        self._maybe_cleanup()
