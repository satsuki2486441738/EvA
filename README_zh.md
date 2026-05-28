<div align="center">

# EvA

### 面向大音频语言模型的 Evidence-First 音频理解范式

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.27667-b31b1b)](https://arxiv.org/abs/2603.27667)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d)](https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)](#安装)

[论文](https://arxiv.org/abs/2603.27667) | [数据集](https://huggingface.co/datasets/SatsukiVie/EvidenceFirst-Audio) | [安装](#安装) | [训练](#训练) | [推理](#推理)

[English](README.md) | 中文

</div>

EvA 为音频语言基础模型加入基于 CED 的第二路音频证据流。共享的 EvA
processor 会将 CED 特征与基础模型的音频 token 对齐，并把融合后的表示注入语言模型。

当前公开版本保留两套支持的骨干：

| Backbone | Package | Training entry |
|---|---|---|
| Kimi-Audio-EvA | `finetune_codes/kimi_audio_eva/` | `src/scripts/train_kimi_audio_eva.sh` |
| Qwen2.5-Omni-EvA | `finetune_codes/qwen2_5_omni_eva/` | `src/scripts/train_qwen2_5_omni_eva.sh` |

共享组件位于 `finetune_codes/eva_processor/`。

## 仓库结构

```text
EvidenceFirst-Audio/
├── finetune_codes/
│   ├── eva_processor/          # 共享 AudioAggregator、CEDProcessor、重采样工具
│   ├── kimi_audio_eva/         # Kimi-Audio-EvA 模型、数据集、LoRA 和导出工具
│   └── qwen2_5_omni_eva/       # Qwen2.5-Omni-EvA 模型、数据集、LoRA 和导出工具
├── src/
│   ├── init/                   # 模型初始化脚本
│   ├── inference/              # 单样本和数据集推理示例
│   ├── scripts/                # 两个独立训练 shell 脚本
│   └── training/               # 面向不同骨干的训练入口
└── demo/                       # 小规模真实音频样例和 smoke checks
```

训练产物、日志、私有数据集、baseline 实验和历史 pipeline 代码不会包含在公开仓库中。

维护约定：

- 每个训练 shell 脚本只启动一个训练任务。
- 骨干模型相关实现放在 `finetune_codes/` 下各自的 package 中。
- 跨骨干共享的融合逻辑只放在 `finetune_codes/eva_processor/`。
- 生成的 checkpoint、日志、私有数据和实验输出不进入版本控制。

## 安装

建议使用 Python 3.10 或 3.11。Python 3.13 暂不推荐用于该栈，因为部分
CUDA/audio 训练依赖可能没有兼容 wheel。

先安装 PyTorch，再以 `--no-build-isolation` 安装 FlashAttention。它们没有固定写入
`requirements.txt`，因为 wheel 和编译参数依赖本机 CUDA 环境。之后安装通用 Python
依赖和本地 package：

```bash
conda create -n eva python=3.10
conda activate eva

# 按你的 CUDA runtime 选择合适的 PyTorch 安装命令。
pip install torch==2.6.0 torchaudio==2.6.0
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 通用 Python 依赖。
pip install -r requirements.txt

pip install -e finetune_codes/eva_processor
pip install -e finetune_codes/kimi_audio_eva
pip install -e finetune_codes/qwen2_5_omni_eva
pip install -e src
```

`src/third_party/kimia_infer` 会保留，因为 Kimi-Audio 推理和 CED 加载依赖它。

开发检查：

```bash
pip install -r requirements-dev.txt
bash demo/scripts/check_demo.sh
python -m compileall -q -x 'src/third_party' .
```

## 准备模型

如果你已经有 EvA-compatible checkpoint，可以直接用于下面的训练或推理命令。

只有当你想从原始上游 backbone 开始并自行创建 EvA base checkpoint 时，才需要运行初始化脚本。
初始化会在微调前加入 EvA/CED 组件。

从原始 Kimi-Audio checkpoint 初始化 Kimi-Audio-EvA：

```bash
python src/init/init_kimi_audio_eva.py \
  --kimi_path /path/to/Kimi-Audio-7B \
  --eva_config_path /path/to/eva-config \
  --output_dir /path/to/Kimi-Audio-EvA-Base
```

从原始 Qwen2.5-Omni checkpoint 初始化 Qwen2.5-Omni-EvA：

```bash
python src/init/init_qwen2_5_omni_eva.py \
  --qwen_omni_path /path/to/Qwen2.5-Omni-7B \
  --ced_path /path/to/ced-base \
  --output_dir /path/to/Qwen2.5-Omni-EvA-Base
```

## 训练

每个 shell 脚本只启动一个 backbone 的一个训练任务。

Kimi-Audio-EvA：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_kimi_audio_eva.sh \
  --model /path/to/Kimi-Audio-EvA-Base \
  --data demo/data/audio_understanding/data_with_semantic_codes.jsonl \
  --output outputs/kimi_audio_eva_demo
```

Qwen2.5-Omni-EvA：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_qwen2_5_omni_eva.sh \
  --model /path/to/Qwen2.5-Omni-EvA-Base \
  --data demo/data/audio_understanding/data.jsonl \
  --output outputs/qwen2_5_omni_eva_demo
```

公开训练脚本暂不启用 DeepSpeed。多卡训练请通过 `CUDA_VISIBLE_DEVICES` 选择 GPU，并使用常规
`torchrun`/DDP。

## 推理

Kimi-Audio-EvA 单条音频：

```bash
python src/inference/infer_kimi_audio_single.py \
  --model_path /path/to/exported-kimi-eva \
  --audio demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```

Kimi-Audio-EvA JSONL 数据集：

```bash
python src/inference/infer_kimi_audio_dataset.py \
  --model_path /path/to/exported-kimi-eva \
  --input_file demo/data/audio_understanding/data.jsonl \
  --output_file outputs/kimi_predictions.jsonl
```

Qwen2.5-Omni-EvA 单条音频：

```bash
python src/inference/infer_qwen2_5_omni_eva.py \
  --model_path /path/to/Qwen2.5-Omni-EvA-Base \
  --audio_path demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```

## 数据格式

Kimi-Audio-EvA 训练数据为 JSONL，每行一个 JSON object：

```json
{"task_type":"understanding","conversation":[{"role":"user","message_type":"text","content":"Please transcribe the spoken content into written text."},{"role":"user","message_type":"audio","content":"audios/example.flac","audio_tokens":[158030,153373]},{"role":"assistant","message_type":"text","content":"transcription text"}]}
```

Qwen2.5-Omni-EvA 可以使用相同格式，但不需要 `audio_tokens`。相对音频路径会按 JSONL
文件所在目录解析。

Kimi-Audio-EvA 的 `audio_tokens` 需要在训练前由 Kimi/GLM-4 audio tokenizer 生成。
从只包含音频路径的 JSONL 开始，运行：

```bash
python -m kimi_audio_eva.extract_semantic_codes \
  --model_name_or_path /path/to/Kimi-Audio-EvA-Base \
  --input_file demo/data/audio_understanding/data.jsonl \
  --output_file demo/data/audio_understanding/data_with_semantic_codes.jsonl
```

这个脚本会给每条 audio message 追加 `audio_tokens` 字段。它支持断点续跑；在可见多 GPU
环境下也可以并行处理。

## 验证

公开仓库包含不依赖模型权重的轻量 smoke tests：

```bash
bash demo/scripts/check_demo.sh
pytest demo/tests
bash src/scripts/train_kimi_audio_eva.sh --help
bash src/scripts/train_qwen2_5_omni_eva.sh --help
```

完整训练和推理需要外部 backbone 与 CED 模型权重。

## 引用

如果 EvA 对你的工作有帮助，请引用：

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

## 许可证

EvA 自有代码使用 [MIT License](LICENSE) 发布。

`src/third_party/` 下随仓库包含的第三方源码不会被 EvA 重新授权。当前仓库中可见的第三方
license 依据见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本仓库不包含模型权重。外部 backbone、tokenizer、CED、Whisper、Kimi-Audio、
Qwen2.5-Omni、GLM-4-Voice 以及 EvA checkpoint 权重需要按各自发布渠道的条款使用。
