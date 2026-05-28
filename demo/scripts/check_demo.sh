#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/finetune_codes:$REPO_ROOT/src:$REPO_ROOT/src/third_party:${PYTHONPATH:-}"

cd "$REPO_ROOT"
python -m compileall -q -x 'src/third_party' .
pytest demo/tests
bash src/scripts/train_kimi_audio_eva.sh --help >/dev/null
bash src/scripts/train_qwen2_5_omni_eva.sh --help >/dev/null

echo "Demo checks passed."
