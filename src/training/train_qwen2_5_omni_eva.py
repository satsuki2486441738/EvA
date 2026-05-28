# train_qwen2_5_omni_eva.py
# Qwen2.5-Omni-EvA training entry.
#
# By default only ced_processor is trainable. Enable LoRA explicitly when the
# downstream task also needs language-model adaptation.

from dataclasses import dataclass, field
import logging
import os
import sys

# ensure third_party (kimia_infer) is importable regardless of cwd
_SRC_ROOT = os.path.dirname(os.path.dirname(__file__))
_THIRD_PARTY = os.path.join(_SRC_ROOT, "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import torch
import transformers
from transformers import Trainer, TrainerCallback, Qwen2_5OmniProcessor
from transformers.trainer_pt_utils import LabelSmoother
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IGNORE_INDEX = LabelSmoother.ignore_index

local_rank = None


def rank0_print(*args):
    if local_rank in (0, None):
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Qwen2.5-Omni-EvA base model path"})
    ced_model_path: str = field(
        default="",
        metadata={"help": "Optional CED model path. Defaults to model_name_or_path/ced-base."},
    )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Training data path (jsonl)"})
    eval_ratio: float = field(default=0.0, metadata={"help": "Evaluation data ratio"})
    max_length: int = field(default=8192, metadata={"help": "Maximum sequence length"})


# 直接使用 transformers.TrainingArguments，不继承
# 这样可以避免 argparse 的格式化字符串问题


@dataclass
class LoRAArguments:
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA for LLM fine-tuning"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "LoRA target layer types (comma-separated)"},
    )
    trainable_modules: Optional[str] = field(
        default="ced_processor",
        metadata={"help": "Full-param trainable module prefixes (comma-separated)"},
    )


@dataclass
class ExportArguments:
    export_split_every_n_epochs: int = field(default=1, metadata={"help": "Export checkpoint every N epochs"})
    export_split_keep_last_k: Optional[int] = field(default=5, metadata={"help": "Keep last K checkpoints"})
    export_split_base_dir: Optional[str] = field(default=None, metadata={"help": "Base directory for checkpoint exports"})


# =========================================================================
# Trainer
# =========================================================================

class AlphaLogCallback(TrainerCallback):
    """Unused placeholder — alpha logging moved into Qwen2_5OmniEvaTrainer.log."""
    pass


class Qwen2_5OmniEvaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Qwen2_5OmniEvaForConditionalGeneration.forward has **kwargs for
        # compatibility, but compute_loss ignores num_items_in_batch. Without
        # this guard, Transformers 5.x assumes the model already normalizes the
        # loss across gradient accumulation and skips its own division.
        self.model_accepts_loss_kwargs = False

    def _find_alpha_name_and_param(self):
        base = self.model
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
            return name, p
        return None, None

    def _find_alpha(self):
        _, p = self._find_alpha_name_and_param()
        return p

    def log(self, logs, start_time=None):
        # piggyback on every log call to print alpha (rank0 only)
        if getattr(self.args, "process_index", 0) == 0 and "loss" in logs:
            name, p = self._find_alpha_name_and_param()
            if p is not None:
                val = float(p.detach().cpu().reshape(()))
                logs["alpha"] = round(val, 6)
                rank0_print(f"[alpha] step={self.state.global_step}  {val:.6f}  ({name})")
        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        # When all audio in a batch fails to load, ced_processor is bypassed and
        # loss has no grad_fn (only frozen embed_tokens participated). Add a dummy
        # zero contribution so DDP backward doesn't crash.
        if loss is not None and loss.grad_fn is None:
            dummy = sum(0.0 * p.sum() for p in model.parameters() if p.requires_grad)
            loss = loss.detach() + dummy
        return (loss, outputs) if return_outputs else loss


# =========================================================================
# Main
# =========================================================================

def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments, LoRAArguments, ExportArguments)
    )
    model_args, data_args, training_args, lora_args, export_args = (
        parser.parse_args_into_dataclasses()
    )
    training_args.remove_unused_columns = False
    local_rank = training_args.local_rank

    # ── load model ──────────────────────────────────────────────────────
    from qwen2_5_omni_eva.modeling_qwen2_5_omni_eva import Qwen2_5OmniEvaForConditionalGeneration
    from qwen2_5_omni_eva.configuration_qwen2_5_omni_eva import Qwen2_5OmniEvaConfig

    rank0_print(f"Loading model from: {model_args.model_name_or_path}")
    config = Qwen2_5OmniEvaConfig.from_pretrained(model_args.model_name_or_path)
    model = Qwen2_5OmniEvaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config,
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    )

    # ── load CED model ──────────────────────────────────────────────────
    ced_path = os.path.join(model_args.model_name_or_path, "ced-base")
    if not os.path.exists(ced_path):
        ced_path = model_args.ced_model_path
    if not ced_path or not os.path.exists(ced_path):
        raise FileNotFoundError(
            "CED model not found. Put it under model_name_or_path/ced-base or pass --ced_model_path."
        )
    rank0_print(f"Loading CED model from: {ced_path}")

    from kimia_infer.models.tokenizer.ced_base.modeling_ced import CedEncoder
    # CED 加载需要访问 HuggingFace，暂时允许在线访问
    old_offline = os.environ.get('HF_HUB_OFFLINE')
    old_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    try:
        ced_model = CedEncoder(ced_path)
    finally:
        # 恢复原始设置
        if old_offline is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = old_offline
        if old_transformers_offline is None:
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
        else:
            os.environ['TRANSFORMERS_OFFLINE'] = old_transformers_offline
    model.set_ced_model(ced_model)

    # ── load processor ──────────────────────────────────────────────────
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or 151643

    # ── data ────────────────────────────────────────────────────────────
    from qwen2_5_omni_eva.datasets import make_qwen2_5_omni_eva_data_module, Qwen2_5OmniEvaDataCollator
    data_module = make_qwen2_5_omni_eva_data_module(
        data_path=data_args.data_path,
        processor=processor,
        max_length=data_args.max_length or training_args.model_max_length,
        eval_ratio=data_args.eval_ratio,
        seed=training_args.seed,
    )

    # ── freeze all params ───────────────────────────────────────────────
    for _, p in model.named_parameters():
        p.requires_grad = False

    # ── LoRA or full-param unfreeze ─────────────────────────────────────
    trainable_mods = [m.strip() for m in lora_args.trainable_modules.split(",") if m.strip()] \
        if lora_args.trainable_modules else []

    if lora_args.use_lora:
        from qwen2_5_omni_eva.lora_utils import attach_lora, print_trainable_params
        model = attach_lora(model, lora_args)
    else:
        from qwen2_5_omni_eva.lora_utils import print_trainable_params
        if not trainable_mods:
            logger.warning("use_lora=False and trainable_modules is empty!")
        for n, p in model.named_parameters():
            if any(n.startswith(pref) or (f".{pref}." in n) for pref in trainable_mods):
                p.requires_grad = True

    # Force-freeze CED encoder (defensive against trainable_modules misuse)
    for n, p in model.named_parameters():
        if ".ced_model." in n or n.startswith("ced_model."):
            p.requires_grad = False

    try:
        model.floating_point_ops = lambda inputs: 0
    except Exception:
        pass

    print_trainable_params(model)

    # ── trainer ─────────────────────────────────────────────────────────
    from qwen2_5_omni_eva.lora_utils import ExportSplitCallback, SaveCheckpointCallback
    collator = Qwen2_5OmniEvaDataCollator(pad_token_id=pad_id)

    trainer = Qwen2_5OmniEvaTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.label_names = ["labels"]
    trainer.add_callback(AlphaLogCallback())

    split_base = export_args.export_split_base_dir or os.path.join(training_args.output_dir, "ckpts")
    trainer.add_callback(ExportSplitCallback(
        export_base_dir=split_base,
        every_n_epochs=export_args.export_split_every_n_epochs,
        keep_last_k=export_args.export_split_keep_last_k,
    ))

    # Save checkpoint every N steps for crash recovery (uses HF native save_steps)
    # training_args.save_steps controls this; SaveCheckpointCallback is a fallback
    # if save_steps is not set via CLI
    if training_args.save_steps <= 0 or training_args.save_steps >= 9999:
        trainer.add_callback(SaveCheckpointCallback(save_steps=200))

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    try:
        trainer.save_state()
    except Exception:
        pass


if __name__ == "__main__":
    train()
