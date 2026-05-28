# Contributing

Thanks for contributing to EvA. This public repository keeps the supported
surface intentionally small: shared EvA processor code, Kimi-Audio-EvA,
Qwen2.5-Omni-EvA, initialization scripts, inference examples and atomic
training launchers.

## Development Setup

Use Python 3.10 or 3.11. Install PyTorch first, then install dependencies and
editable local packages:

```bash
pip install torch==2.6.0 torchaudio==2.6.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r src/requirements.txt
pip install -e finetune_codes/eva_processor
pip install -e finetune_codes/kimi_audio_eva
pip install -e finetune_codes/qwen2_5_omni_eva
pip install -e src
```

## Checks

Run the lightweight checks before opening a pull request:

```bash
python -m compileall -q -x 'src/third_party' .
bash demo/scripts/check_demo.sh
pytest demo/tests
bash src/scripts/train_kimi_audio_eva.sh --help
bash src/scripts/train_qwen2_5_omni_eva.sh --help
```

Numerical training and inference require external model weights and are not
expected to run in CI.

## Repository Boundaries

- Do not commit checkpoints, model weights, logs, private datasets or generated
  experiment outputs.
- Keep backbone-specific changes inside the matching package under
  `finetune_codes/`.
- Keep shared audio-fusion logic in `finetune_codes/eva_processor/`.
- Keep training launchers atomic: one script should launch one training job.
