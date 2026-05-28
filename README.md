<div align="center">

# EvA

### An Evidence-First Audio Understanding Paradigm for LALMs

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.27667-b31b1b)](https://arxiv.org/abs/2603.27667)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d)](https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)](#installation)

[Paper](https://arxiv.org/abs/2603.27667) | [Dataset](https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio) | [Models](#resources) | [Installation](#installation) | [Training](#train) | [Inference](#inference)

English | [中文](README_zh.md)

</div>

EvA adds a CED-based second audio stream to audio-language foundation models.
The shared EvA processor aligns CED features with the base model audio tokens and
injects the fused representation into the language model.

## Highlights

| Item | Description |
|---|---|
| Evidence-first audio modeling | Adds an explicit CED evidence stream before the language model consumes fused audio representations. |
| Two supported backbones | Public code supports Kimi-Audio-EvA and Qwen2.5-Omni-EvA. |
| Open-source runnable layout | Includes install instructions, demo data, smoke tests, training scripts and inference entry points. |
| Clean public scope | Checkpoints, logs, private datasets and historical experiment outputs are intentionally excluded. |

## Resources

| Resource | Link |
|---|---|
| Paper | [arXiv:2603.27667](https://arxiv.org/abs/2603.27667) |
| Dataset | [Hugging Face: SatsukiVie/EvidenceFirst-Audio](https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio) |
| Models | Hugging Face: TBD; ModelScope: TBD |
| Dataset mirror | ModelScope: TBD |

This public version keeps two supported backbones:

| Backbone | Package | Training entry |
|---|---|---|
| Kimi-Audio-EvA | `finetune_codes/kimi_audio_eva/` | `src/scripts/train_kimi_audio_eva.sh` |
| Qwen2.5-Omni-EvA | `finetune_codes/qwen2_5_omni_eva/` | `src/scripts/train_qwen2_5_omni_eva.sh` |

The shared components live in `finetune_codes/eva_processor/`.

## Repository Layout

```text
EvidenceFirst-Audio/
├── finetune_codes/
│   ├── eva_processor/          # Shared AudioAggregator, CEDProcessor, resampling utilities
│   ├── kimi_audio_eva/         # Kimi-Audio-EvA model, dataset, LoRA and export utilities
│   └── qwen2_5_omni_eva/       # Qwen2.5-Omni-EvA model, dataset, LoRA and export utilities
├── src/
│   ├── init/                   # Model initialization scripts
│   ├── inference/              # Single and dataset inference examples
│   ├── scripts/                # Two atomic training shell scripts
│   └── training/               # Backbone-specific training entries
└── demo/                       # Real small audio examples and smoke checks
```

Training products, logs, private datasets, baseline experiments and historical
pipeline code are intentionally not included.

Maintenance rules for this public tree:

- Keep each training shell script atomic: one script launches one training job.
- Keep backbone-specific implementation inside its package under `finetune_codes/`.
- Put cross-backbone fusion logic only in `finetune_codes/eva_processor/`.
- Keep generated checkpoints, logs, private datasets and experiment outputs out
  of source control.

## Installation

Use Python 3.10 or 3.11. Python 3.13 is not recommended for this stack because
some CUDA/audio training dependencies may not provide compatible wheels.

Install PyTorch first, then install FlashAttention without build isolation.
Those packages are intentionally not pinned inside `requirements.txt` because
their wheels and build flags depend on the local CUDA runtime. After that,
install the remaining Python dependencies and local packages:

```bash
conda create -n eva python=3.10
conda activate eva

# Choose the PyTorch command that matches your CUDA runtime.
pip install torch==2.6.0 torchaudio==2.6.0
pip install flash-attn==2.7.4.post1 --no-build-isolation

# General Python dependencies.
pip install -r requirements.txt

pip install -e finetune_codes/eva_processor
pip install -e finetune_codes/kimi_audio_eva
pip install -e finetune_codes/qwen2_5_omni_eva
pip install -e src
```

`src/third_party/kimia_infer` is kept because Kimi-Audio inference and CED loading
depend on it.

For development-only checks:

```bash
pip install -r requirements-dev.txt
bash demo/scripts/check_demo.sh
python -m compileall -q -x 'src/third_party' .
```

## Prepare Models

If you already have an EvA-compatible checkpoint, you can use it directly with
the training or inference commands below.

Run the initialization scripts only when you want to start from an original
upstream backbone and create an EvA base checkpoint yourself. The initialization
step adds the EvA/CED components before fine-tuning.

Initialize Kimi-Audio-EvA from an original Kimi-Audio checkpoint:

```bash
python src/init/init_kimi_audio_eva.py \
  --kimi_path /path/to/Kimi-Audio-7B \
  --eva_config_path /path/to/eva-config \
  --output_dir /path/to/Kimi-Audio-EvA-Base
```

Initialize Qwen2.5-Omni-EvA from an original Qwen2.5-Omni checkpoint:

```bash
python src/init/init_qwen2_5_omni_eva.py \
  --qwen_omni_path /path/to/Qwen2.5-Omni-7B \
  --ced_path /path/to/ced-base \
  --output_dir /path/to/Qwen2.5-Omni-EvA-Base
```

## Train

Each shell script launches exactly one training job for one backbone.

Kimi-Audio-EvA:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_kimi_audio_eva.sh \
  --model /path/to/Kimi-Audio-EvA-Base \
  --data demo/data/audio_understanding/data_with_semantic_codes.jsonl \
  --output outputs/kimi_audio_eva_demo
```

Qwen2.5-Omni-EvA:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_qwen2_5_omni_eva.sh \
  --model /path/to/Qwen2.5-Omni-EvA-Base \
  --data demo/data/audio_understanding/data.jsonl \
  --output outputs/qwen2_5_omni_eva_demo
```

DeepSpeed is not enabled in the public training scripts. Use regular
`torchrun`/DDP by selecting one or more GPUs with `CUDA_VISIBLE_DEVICES`.

## Inference

Kimi-Audio-EvA single audio:

```bash
python src/inference/infer_kimi_audio_single.py \
  --model_path /path/to/exported-kimi-eva \
  --audio demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```

Kimi-Audio-EvA JSONL dataset:

```bash
python src/inference/infer_kimi_audio_dataset.py \
  --model_path /path/to/exported-kimi-eva \
  --input_file demo/data/audio_understanding/data.jsonl \
  --output_file outputs/kimi_predictions.jsonl
```

Qwen2.5-Omni-EvA single audio:

```bash
python src/inference/infer_qwen2_5_omni_eva.py \
  --model_path /path/to/Qwen2.5-Omni-EvA-Base \
  --audio_path demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```

## Data Format

Kimi-Audio-EvA training expects one JSON object per line:

```json
{"task_type":"understanding","conversation":[{"role":"user","message_type":"text","content":"Please transcribe the spoken content into written text."},{"role":"user","message_type":"audio","content":"audios/example.flac","audio_tokens":[158030,153373]},{"role":"assistant","message_type":"text","content":"transcription text"}]}
```

Qwen2.5-Omni-EvA can use the same format without `audio_tokens`. Relative audio
paths are resolved relative to the JSONL file location.

## Validation

The public repository includes lightweight smoke tests that do not require model
weights:

```bash
bash demo/scripts/check_demo.sh
pytest demo/tests
bash src/scripts/train_kimi_audio_eva.sh --help
bash src/scripts/train_qwen2_5_omni_eva.sh --help
```

Full training and inference require external backbone and CED model weights.

## Citation

If you find EvA useful, please cite:

```bibtex
@misc{xie2026evaevidencefirstaudiounderstanding,
      title={EvA: An Evidence-First Audio Understanding Paradigm for LALMs}, 
      author={Xinyuan Xie and Shunian Chen and Zhiheng Liu and Yuhao Zhang and Zhiqiang Lv and Liyin Liang and Benyou Wang},
      year={2026},
      eprint={2603.27667},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.27667}, 
}
```

## License

EvA's own code is released under the [MIT License](LICENSE).

Third-party source code included under `src/third_party/` is not relicensed by
EvA. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the license
evidence currently present in this repository.

Model weights are not included in this repository. External backbone, tokenizer,
CED, Whisper, Kimi-Audio, Qwen2.5-Omni, GLM-4-Voice, and EvA checkpoint weights
must be used under their own distribution terms.
