# Kimi-Audio-EvA Package

This package contains Kimi-Audio-EvA model integration, dataset processing, LoRA
helpers and export utilities.

## Demo Data

The repository-level `demo/data/audio_understanding/` directory contains a few
LibriSpeech-style samples:

- `data.jsonl`: text + audio paths, usable by Qwen2.5-Omni-EvA examples.
- `data_with_semantic_codes.jsonl`: the same samples with Kimi semantic audio
  tokens, usable by Kimi-Audio-EvA training examples.
- `audios/`: short FLAC audio files referenced by the JSONL files.

Audio paths in the JSONL files are relative to the JSONL location.

## Data Format

```json
{
  "task_type": "understanding",
  "conversation": [
    {"role": "user", "message_type": "text", "content": "Please transcribe the spoken content into written text."},
    {"role": "user", "message_type": "audio", "content": "audios/example.flac", "audio_tokens": [158030, 153373]},
    {"role": "assistant", "message_type": "text", "content": "transcription text"}
  ]
}
```

`audio_tokens` are required for Kimi-Audio-EvA training. They can be produced with
`kimi_audio_eva.extract_semantic_codes` when a compatible Kimi-Audio tokenizer is
available.

## Training

Use the repository-level atomic training launcher:

```bash
bash src/scripts/train_kimi_audio_eva.sh \
  --model /path/to/Kimi-Audio-EvA-Base \
  --data demo/data/audio_understanding/data_with_semantic_codes.jsonl \
  --output outputs/kimi_audio_eva_demo
```
