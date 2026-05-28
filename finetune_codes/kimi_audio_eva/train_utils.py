# kimi_audio_eva/train_utils.py
# Shared dataclasses and data-loading utilities for finetune entry scripts.
from dataclasses import dataclass, field
import json
import os
import random
from typing import Dict, Optional

from .datasets import LazySupervisedDataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B")
    model_path: Optional[str] = field(default=None, metadata={"help": "Local pretrained path (optional)."})
    use_ced_in_forward: bool = field(default=True, metadata={"help": "Whether to use CED features in forward pass (for ablation study)."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Training data path (jsonl, one sample per line)."})
    eval_ratio: float = field(default=0.0, metadata={"help": "Validation split ratio; 0 to disable."})
    lazy_preprocess: bool = False
    skip_audio_check: bool = False


@dataclass
class ExportArguments:
    export_split_base_dir: Optional[str] = field(default=None, metadata={"help": "Base dir for split checkpoints; default=output_dir/split_ckpts."})
    export_split_every_n_epochs: int = field(default=1)
    export_split_keep_last_k: Optional[int] = field(default=3)


def make_supervised_data_module(text_tokenizer, data_args, max_len, kimia_token_offset,
                                seed: int = 42) -> Dict:
    print("Loading data...")
    data_dir = os.path.dirname(os.path.abspath(data_args.data_path))
    with open(data_args.data_path, "r") as f:
        all_data = [json.loads(line) for line in f]
    for item in all_data:
        for message in item.get("conversation", []):
            if message.get("message_type") == "audio":
                content = message.get("content")
                if content and not os.path.isabs(content):
                    message["content"] = os.path.normpath(os.path.join(data_dir, content))
    random.seed(seed)
    random.shuffle(all_data)
    if data_args.eval_ratio and data_args.eval_ratio > 0:
        ev_n = int(len(all_data) * data_args.eval_ratio)
        eval_data, train_data = all_data[:ev_n], all_data[ev_n:]
        assert len(eval_data) > 0 and len(train_data) > 0
    else:
        eval_data, train_data = None, all_data

    train_dataset = LazySupervisedDataset(
        train_data, text_tokenizer=text_tokenizer, max_len=max_len,
        kimia_token_offset=kimia_token_offset,
        skip_audio_check=data_args.skip_audio_check,
    )
    eval_dataset = LazySupervisedDataset(
        eval_data, text_tokenizer=text_tokenizer, max_len=max_len,
        kimia_token_offset=kimia_token_offset,
        skip_audio_check=data_args.skip_audio_check,
    ) if eval_data else None

    print(f"Train dataset length: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset length: {len(eval_dataset)}")

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
