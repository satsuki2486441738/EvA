# Source Entrypoints

`src/` contains project-level executable code only. Backbone-specific model,
dataset, LoRA and export code lives under `finetune_codes/`.

```text
src/
├── training/
│   ├── train_kimi_audio_eva.py
│   └── train_qwen2_5_omni_eva.py
├── init/
│   ├── common.py
│   ├── init_kimi_audio_eva.py
│   └── init_qwen2_5_omni_eva.py
├── inference/
│   ├── infer_kimi_audio_single.py
│   ├── infer_kimi_audio_dataset.py
│   └── infer_qwen2_5_omni_eva.py
├── scripts/
│   ├── train_kimi_audio_eva.sh
│   └── train_qwen2_5_omni_eva.sh
└── third_party/
    └── kimia_infer/
```

## Atomic Training Scripts

Install runtime dependencies from the repository root with
`pip install -r requirements.txt`. There is intentionally no separate
`src/requirements.txt`; `src/` only contains executable entrypoints and the
vendored `kimia_infer` package metadata.

Kimi-Audio-EvA:

```bash
bash src/scripts/train_kimi_audio_eva.sh \
  --model /path/to/Kimi-Audio-EvA-Base \
  --data /path/to/data_with_semantic_codes.jsonl \
  --output /path/to/output
```

Qwen2.5-Omni-EvA:

```bash
bash src/scripts/train_qwen2_5_omni_eva.sh \
  --model /path/to/Qwen2.5-Omni-EvA-Base \
  --data /path/to/data.jsonl \
  --output /path/to/output
```

The shell scripts do not run preprocessing, evaluation, checkpoint conversion or
multi-step task chains.

## Inference Entrypoints

```bash
python src/inference/infer_kimi_audio_single.py --model_path /path/to/model --audio /path/to/audio.wav
python src/inference/infer_kimi_audio_dataset.py --model_path /path/to/model --input_file /path/to/data.jsonl --output_file /path/to/predictions.jsonl
python src/inference/infer_qwen2_5_omni_eva.py --model_path /path/to/model --audio_path /path/to/audio.wav
```
