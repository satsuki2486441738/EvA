# lora_utils.py
import os
import gc
import types
import inspect
import logging
from typing import List, Optional

import torch
from torch import nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


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
    rank0_print(f"Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.3f}M ({100*trainable/total:.2f}%)")
    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(n, p.shape)


def _unwrap_ddp(model):
    return getattr(model, "module", model)


def _collect_trainable_non_lora_state_dict(model):
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            state[name] = param.detach().cpu()
    return state


def collect_lora_targets(
    model: nn.Module,
    lora_modules: Optional[List[str]] = None,
    lora_target_layers: Optional[List[str]] = None,
    excluded_prefixes: Optional[List[str]] = None,
) -> List[str]:
    excluded = list(excluded_prefixes or []) + ["audio_tower", "ced_model", "ced_processor"]

    def _excluded(name):
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


def _ensure_prepare_inputs_for_generation(model):
    if hasattr(model, "prepare_inputs_for_generation"):
        return model

    def _pifg(self, **kwargs):
        return {k: v for k, v in kwargs.items()}

    model.prepare_inputs_for_generation = types.MethodType(_pifg, model)
    return model


def _patch_peft_forward(peft_model):
    base = peft_model.get_base_model()
    try:
        allowed = set(inspect.signature(base.forward).parameters.keys())
    except Exception:
        allowed = set()
    allowed |= {
        "use_cache", "output_attentions", "output_hidden_states", "return_dict",
        "position_ids", "attention_mask", "past_key_values", "inputs_embeds",
        "cache_position", "input_features", "feature_attention_mask",
        "audio_feature_lengths", "audio_waveform", "audio_mask",
        "audio_waveform_lengths", "image_grid_thw", "video_grid_thw",
        "use_audio_in_video", "video_second_per_grid", "labels",
    }

    def _forward(self, *args, **kwargs):
        fkwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return self.base_model(*args, **fkwargs)

    peft_model.forward = types.MethodType(_forward, peft_model)


def attach_lora(model, lora_args):
    lora_target_layers = [l.strip() for l in lora_args.lora_target_modules.split(",") if l.strip()] \
        if getattr(lora_args, "lora_target_modules", None) else None
    trainable_mods = [m.strip() for m in lora_args.trainable_modules.split(",") if m.strip()] \
        if getattr(lora_args, "trainable_modules", None) else []

    model = _ensure_prepare_inputs_for_generation(model)

    targets = collect_lora_targets(
        model, lora_modules=["model.layers"],
        lora_target_layers=lora_target_layers,
        excluded_prefixes=trainable_mods,
    )
    if not targets:
        raise RuntimeError("No LoRA target modules found.")

    module_names = set(dict(model.named_modules()).keys())
    modules_to_save = [m for m in trainable_mods if m in module_names]

    lconf = LoraConfig(
        r=getattr(lora_args, "lora_r", 16),
        lora_alpha=getattr(lora_args, "lora_alpha", 32),
        lora_dropout=getattr(lora_args, "lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
        modules_to_save=modules_to_save or None,
    )
    peft_model = get_peft_model(model, lconf, adapter_name="default")
    _patch_peft_forward(peft_model)

    for _, p in peft_model.named_parameters():
        p.requires_grad = False
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    if trainable_mods:
        for n, p in peft_model.named_parameters():
            clean_name = n
            for prefix in ("base_model.model.", "model."):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            if "lora_" not in n and any(clean_name.startswith(pref) for pref in trainable_mods):
                p.requires_grad = True
    return peft_model


def export_split_from_model(model_or_wrapper, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    base = _unwrap_ddp(model_or_wrapper)

    if isinstance(base, PeftModel):
        # Do not call merge_and_unload() on the live training model here. It
        # mutates the PEFT wrapper and can silently break later epochs.
        base.save_pretrained(export_dir, safe_serialization=True)
        merged = base.get_base_model()
        trainable_non_lora = _collect_trainable_non_lora_state_dict(base)
        if trainable_non_lora:
            torch.save(trainable_non_lora, os.path.join(export_dir, "trainable_non_lora.pt"))
        ced_processor = getattr(merged, "ced_processor", None)
        if ced_processor is not None:
            torch.save(ced_processor.state_dict(), os.path.join(export_dir, "ced_processor.pt"))
        if getattr(merged, "config", None) is not None:
            merged.config.save_pretrained(export_dir)
    else:
        merged = base
        ced_model = getattr(merged, "ced_model", None)
        had_ced_model = hasattr(merged, "ced_model")
        if had_ced_model:
            merged.ced_model = None
        try:
            merged.save_pretrained(export_dir, safe_serialization=True, max_shard_size="5GB")
        finally:
            if had_ced_model:
                merged.ced_model = ced_model

    # copy ced-base and processor files from original model path
    base_path = getattr(getattr(merged, "config", None), "_name_or_path", None)
    if base_path and os.path.isdir(base_path):
        import shutil
        ced_src = os.path.join(base_path, "ced-base")
        ced_dst = os.path.join(export_dir, "ced-base")
        if os.path.exists(ced_src) and not os.path.exists(ced_dst):
            shutil.copytree(ced_src, ced_dst)
        # copy processor/tokenizer files so the checkpoint is self-contained
        processor_files = [
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ]
        for fname in processor_files:
            src = os.path.join(base_path, fname)
            dst = os.path.join(export_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    logger.info(f"Exported model to: {export_dir}")
    return export_dir


def load_adapter_for_inference(base_model, adapter_dir, merge=False):
    base_model = _ensure_prepare_inputs_for_generation(base_model)
    base_device = None
    try:
        base_device = next(base_model.parameters()).device
    except Exception:
        pass
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    _patch_peft_forward(model)
    extra_path = os.path.join(adapter_dir, "trainable_non_lora.pt")
    if os.path.exists(extra_path):
        extra_state = torch.load(extra_path, map_location=base_device or "cpu")
        missing, unexpected = model.load_state_dict(extra_state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected trainable_non_lora keys: {unexpected[:10]}")
        if missing:
            logger.info(f"Missing keys while loading trainable_non_lora: {len(missing)}")

    ced_path = os.path.join(adapter_dir, "ced_processor.pt")
    target = model.get_base_model()
    if os.path.exists(ced_path) and getattr(target, "ced_processor", None) is not None:
        target.ced_processor.load_state_dict(torch.load(ced_path, map_location=base_device or "cpu"), strict=False)

    if merge:
        model = model.merge_and_unload()
    return model


class SaveCheckpointCallback(TrainerCallback):
    """Save full checkpoint every N steps for crash recovery"""
    def __init__(self, save_steps=100):
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            control.should_save = True
        return control


class ExportSplitCallback(TrainerCallback):
    def __init__(self, export_base_dir, every_n_epochs=1, keep_last_k=None):
        super().__init__()
        self.export_base_dir = export_base_dir
        self.every = int(every_n_epochs)
        self.keep_last_k = keep_last_k
        os.makedirs(self.export_base_dir, exist_ok=True)
        self._epoch_dirs: List[str] = []

    def _barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control
        ep = int(state.epoch)
        # print alpha at every epoch end regardless of export schedule
        if getattr(args, "process_index", 0) == 0:
            model = kwargs.get("model")
            if model is not None:
                base = model
                for _ in range(5):
                    base = getattr(base, "module", base)
                candidates = []
                for name, p in base.named_parameters():
                    if name.endswith("ced_processor.modules_to_save.default.alpha"):
                        candidates.append((0, name, p))
                    elif name.endswith("ced_processor.alpha"):
                        candidates.append((1, name, p))
                    elif name.endswith("ced_processor.original_module.alpha"):
                        candidates.append((2, name, p))
                if candidates:
                    _, name, p = sorted(candidates, key=lambda x: x[0])[0]
                    val = float(p.detach().cpu().reshape(()))
                    rank0_print(f"[alpha] epoch={ep}  {val:.6f}  ({name})")
        if self.every <= 0:
            return control
        if ep % self.every != 0:
            return control
        self._barrier()
        if getattr(args, "process_index", 0) == 0:
            model = kwargs.get("model")
            if model is not None:
                out_dir = os.path.join(self.export_base_dir, f"epoch_{ep:03d}")
                rank0_print(f"[ExportSplit] epoch {ep} -> {out_dir}")
                export_split_from_model(model, out_dir)
                self._epoch_dirs.append(out_dir)
                if self.keep_last_k and len(self._epoch_dirs) > self.keep_last_k:
                    import shutil
                    to_rm = self._epoch_dirs.pop(0)
                    shutil.rmtree(to_rm, ignore_errors=True)
        self._barrier()
        return control
