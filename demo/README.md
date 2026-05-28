# Demo

This directory contains small real audio-understanding examples for smoke tests,
training command examples and inference command examples.

## Data

`data/audio_understanding/` contains LibriSpeech-style FLAC samples:

- `data.jsonl`: text + audio paths. This is usable by Qwen2.5-Omni-EvA examples.
- `data_with_semantic_codes.jsonl`: the same samples with Kimi semantic audio
  tokens. This is usable by Kimi-Audio-EvA examples.
- `audios/`: short FLAC files referenced by the JSONL files.

Audio paths in the JSONL files are relative to the JSONL location.

## Checks

Run the lightweight demo checks:

```bash
bash demo/scripts/check_demo.sh
```

These checks do not require model weights. Full training and inference still
require external backbone and CED model weights.

## Example Commands

Kimi-Audio-EvA training smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_kimi_audio_eva.sh \
  --model /path/to/Kimi-Audio-EvA-Base \
  --data demo/data/audio_understanding/data_with_semantic_codes.jsonl \
  --output outputs/kimi_audio_eva_demo
```

Qwen2.5-Omni-EvA training smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/scripts/train_qwen2_5_omni_eva.sh \
  --model /path/to/Qwen2.5-Omni-EvA-Base \
  --data demo/data/audio_understanding/data.jsonl \
  --output outputs/qwen2_5_omni_eva_demo
```

Kimi-Audio-EvA single-audio inference:

```bash
python src/inference/infer_kimi_audio_single.py \
  --model_path /path/to/exported-kimi-eva \
  --audio demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```

Qwen2.5-Omni-EvA single-audio inference:

```bash
python src/inference/infer_qwen2_5_omni_eva.py \
  --model_path /path/to/Qwen2.5-Omni-EvA-Base \
  --audio_path demo/data/audio_understanding/audios/librispeech_1263-139804-0001.flac
```
