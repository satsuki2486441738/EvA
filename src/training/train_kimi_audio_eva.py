# train_kimi_audio_eva.py
# Kimi-Audio-EvA training entry: supports full-parameter and LoRA modes.
#
# 使用方式：
#   use_lora=False（默认）：仅对 trainable_modules 全参微调
#     torchrun ... train_kimi_audio_eva.py --trainable_modules model.ced_processor,model.vq_adaptor ...
#
#   use_lora=True：对 lora_modules 注入 LoRA，trainable_modules 额外全参
#     torchrun ... train_kimi_audio_eva.py --use_lora True --lora_r 64 --lora_alpha 64 \
#                           --trainable_modules model.ced_processor,model.vq_adaptor ...

from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Optional
from functools import partial

# ensure third_party (kimia_infer) is importable regardless of cwd
_SRC_ROOT = os.path.dirname(os.path.dirname(__file__))
_THIRD_PARTY = os.path.join(_SRC_ROOT, "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import torch
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from huggingface_hub import snapshot_download

from kimi_audio_eva.model import KimiAudioModel
from kimi_audio_eva.modeling_kimia import CED_USE_FREQ_GATE, CED_USE_CROSS_LAYER_FUSION
from kimi_audio_eva.train_utils import ModelArguments, DataArguments, ExportArguments, make_supervised_data_module
from kimi_audio_eva.datasets import LazySupervisedDataset
from kimi_audio_eva.lora_utils import attach_lora, ExportSplitCallback, print_trainable_params
from kimia_infer.utils.special_tokens import instantiate_extra_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None


def rank0_print(*args):
    if local_rank in (0, None):
        print(*args)


# =============================================================================
# Argument dataclasses
# =============================================================================

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    model_max_length: int = field(default=8192)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    # 全参训练 ced_processor/vq_adaptor 时会有未使用参数（消融开关关闭时），需要设为 True
    ddp_find_unused_parameters: bool = field(default=True)
    save_strategy: str = field(default="no")


@dataclass
class LoRAArguments:
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否启用 LoRA。False=仅对 trainable_modules 全参微调"},
    )
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    adapter_name: str = field(default="default")
    lora_modules: Optional[str] = field(
        default="model.layers",
        metadata={"help": "LoRA 作用范围的模块前缀（逗号分隔）。默认: model.layers"},
    )
    lora_target_layers: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "LoRA 目标层类型（逗号分隔）。默认: q_proj,k_proj,v_proj,o_proj"},
    )
    trainable_modules: Optional[str] = field(
        default="model.ced_processor,model.vq_adaptor",
        metadata={
            "help": (
                "全参可训模块前缀（逗号分隔）。"
                "use_lora=False 时作为唯一可训模块；"
                "use_lora=True 时在 LoRA 之外额外全量训练。"
                "默认: model.ced_processor,model.vq_adaptor"
            )
        },
    )


# =============================================================================
# Trainer
# =============================================================================

class KimiAudioTrainer(Trainer):
    """
    统一 Trainer：
    - compute_loss：仅计算文本 CE，配合 datasets.py 已完成的 label shift + mask
    - optimizer_step：每步后 log ced_processor.alpha（若存在）
    """

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        try:
            base = getattr(self.model, "module", self.model)
            for name, p in base.named_parameters():
                if name.endswith("ced_processor.alpha"):
                    val = float(p.detach().cpu().reshape(()))
                    try:
                        self.log({"alpha": val})
                    except Exception:
                        pass
                    try:
                        print(f"[alpha] step={self.state.global_step}  {val:.6f}", flush=True)
                    except Exception:
                        pass
                    break
        except Exception:
            pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        text_logits = outputs.logits
        _, text_labels, _, text_loss_mask = labels
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = ce(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1))
        loss = (loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-6)
        if loss.grad_fn is None:
            dummy = sum(0.0 * p.sum() for p in model.parameters() if p.requires_grad)
            loss = loss.detach() + dummy
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Main
# =============================================================================

def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments, ExportArguments)
    )
    model_args, data_args, training_args, lora_args, export_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    local_rank = training_args.local_rank
    if torch.cuda.is_available() and local_rank not in (-1, None):
        torch.cuda.set_device(local_rank)
        rank0_print(f"[DDP] set CUDA device to local_rank={local_rank}")

    # ── 加载模型与 tokenizer ──────────────────────────────────────────────────
    logger.info("Loading Kimi-Audio base model")
    cache_path = model_args.model_name_or_path if os.path.exists(model_args.model_name_or_path) \
                 else snapshot_download(model_args.model_name_or_path)
    if model_args.model_path and not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    model = KimiAudioModel.init_from_pretrained(
        model_name_or_path=model_args.model_name_or_path,
        model_load_kwargs={"low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16},
    )
    # 设置消融开关：是否在前向传播中使用CED特征
    model.use_ced_in_forward = model_args.use_ced_in_forward
    if not model_args.use_ced_in_forward:
        rank0_print(f"[Ablation] CED features will be MASKED in forward pass (use_ced_in_forward=False)")

    tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
    extra = instantiate_extra_tokens(tokenizer)
    pad_id = extra.pad

    # ── 数据 ─────────────────────────────────────────────────────────────────
    data_module = make_supervised_data_module(
        text_tokenizer=tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model.config.kimia_token_offset,
        seed=training_args.seed,
    )

    # ── 冻结全部参数 ──────────────────────────────────────────────────────────
    for _, p in model.named_parameters():
        p.requires_grad = False

    # ── LoRA 或全参解冻 ───────────────────────────────────────────────────────
    if lora_args.use_lora:
        model = attach_lora(model, lora_args)
    else:
        trainable_mods = [m.strip() for m in lora_args.trainable_modules.split(",") if m.strip()] \
            if lora_args.trainable_modules else []
        if not trainable_mods:
            logger.warning("use_lora=False 且 trainable_modules 未设置，没有任何参数会被训练！")
        for n, p in model.named_parameters():
            if any(n.startswith(pref) for pref in trainable_mods):
                p.requires_grad = True

    # ── 强制冻结 CED encoder / Whisper encoder（防御性：避免 trainable_modules 误开启）─
    for n, p in model.named_parameters():
        if ".ced_model." in n or n.startswith("ced_model."):
            p.requires_grad = False
        if ".whisper_model." in n or n.startswith("whisper_model."):
            p.requires_grad = False

    # ── 消融开关：进一步冻结不参与前向的参数（避免 DDP unused 参数报错）─────────
    if not CED_USE_FREQ_GATE:
        for name, p in model.named_parameters():
            if "model.ced_processor.audio_aggregator.gate" in name:
                p.requires_grad = False

    if not CED_USE_CROSS_LAYER_FUSION:
        for name, p in model.named_parameters():
            if (
                "model.ced_processor.audio_aggregator.aggregator_layer_1" in name
                or "model.ced_processor.audio_aggregator.aggregator_layer_2" in name
            ):
                p.requires_grad = False

    try:
        model.floating_point_ops = lambda inputs: 0
    except Exception:
        pass

    print_trainable_params(model)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = KimiAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=partial(LazySupervisedDataset.collate_fn, pad_token_id=pad_id),
        processing_class=tokenizer,
    )
    trainer.label_names = ["labels"]

    split_base = export_args.export_split_base_dir or os.path.join(training_args.output_dir, "split_ckpts")
    trainer.add_callback(ExportSplitCallback(
        export_base_dir=split_base,
        every_n_epochs=export_args.export_split_every_n_epochs,
        keep_last_k=export_args.export_split_keep_last_k,
    ))

    trainer.train()
    try:
        trainer.save_state()
    except Exception:
        pass


if __name__ == "__main__":
    train()
