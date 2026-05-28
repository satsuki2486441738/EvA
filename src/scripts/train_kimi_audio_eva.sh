#!/usr/bin/env bash
# Atomic training entry for Kimi-Audio-EvA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$REPO_ROOT/finetune_codes:$REPO_ROOT/src:$REPO_ROOT/src/third_party:${PYTHONPATH:-}"

MODEL_PATH=""
DATA_PATH=""
OUTPUT_DIR=""
LOG_FILE=""

NUM_EPOCHS=1
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
MODEL_MAX_LENGTH=512
LEARNING_RATE=1e-3
TRAINABLE_MODULES=model.ced_processor
USE_LORA=False
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.05
EVAL_RATIO=0.0
MASTER_PORT="${MASTER_PORT:-6001}"
DATALOADER_NUM_WORKERS=4

usage() {
  cat <<USAGE
Usage: $(basename "$0") --model MODEL_PATH --data DATA_PATH --output OUTPUT_DIR [options]

Required:
  --model, -m PATH              Kimi-Audio-EvA model directory.
  --data, -d PATH               Training JSONL path.
  --output, -o PATH             Output checkpoint directory.

Options:
  --epochs N                    Number of epochs. Default: $NUM_EPOCHS
  --batch-size N                Per-device train batch size. Default: $PER_DEVICE_BATCH_SIZE
  --grad-accum N                Gradient accumulation steps. Default: $GRADIENT_ACCUMULATION_STEPS
  --max-length N                Model max length. Default: $MODEL_MAX_LENGTH
  --lr VALUE                    Learning rate. Default: $LEARNING_RATE
  --trainable-modules VALUE     Comma-separated trainable module prefixes. Default: $TRAINABLE_MODULES
  --use-lora true|false         Enable LoRA. Default: $USE_LORA
  --lora-r N                    LoRA rank. Default: $LORA_R
  --lora-alpha N                LoRA alpha. Default: $LORA_ALPHA
  --lora-dropout VALUE          LoRA dropout. Default: $LORA_DROPOUT
  --eval-ratio VALUE            Evaluation split ratio. Default: $EVAL_RATIO
  --log-file PATH               Log file. Default: OUTPUT_DIR/train.log
  --workers N                   Dataloader workers. Default: $DATALOADER_NUM_WORKERS
  --help, -h                    Show this help.

Environment:
  CUDA_VISIBLE_DEVICES=0,1      Select visible GPUs before running this script.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m) shift; MODEL_PATH="${1:-}" ;;
    --data|-d) shift; DATA_PATH="${1:-}" ;;
    --output|-o) shift; OUTPUT_DIR="${1:-}" ;;
    --epochs|-e) shift; NUM_EPOCHS="${1:-}" ;;
    --batch-size) shift; PER_DEVICE_BATCH_SIZE="${1:-}" ;;
    --grad-accum) shift; GRADIENT_ACCUMULATION_STEPS="${1:-}" ;;
    --max-length) shift; MODEL_MAX_LENGTH="${1:-}" ;;
    --lr) shift; LEARNING_RATE="${1:-}" ;;
    --trainable-modules) shift; TRAINABLE_MODULES="${1:-}" ;;
    --use-lora) shift; USE_LORA="${1:-}" ;;
    --lora-r) shift; LORA_R="${1:-}" ;;
    --lora-alpha) shift; LORA_ALPHA="${1:-}" ;;
    --lora-dropout) shift; LORA_DROPOUT="${1:-}" ;;
    --eval-ratio) shift; EVAL_RATIO="${1:-}" ;;
    --log-file) shift; LOG_FILE="${1:-}" ;;
    --workers) shift; DATALOADER_NUM_WORKERS="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ -z "$MODEL_PATH" || -z "$DATA_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required argument."
  usage
  exit 1
fi

[[ -d "$MODEL_PATH" ]] || { echo "Model path not found: $MODEL_PATH"; exit 1; }
[[ -f "$DATA_PATH" ]] || { echo "Data path not found: $DATA_PATH"; exit 1; }

MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
DATA_PATH="$(cd "$(dirname "$DATA_PATH")" && pwd)/$(basename "$DATA_PATH")"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$OUTPUT_DIR/train.log"
else
  LOG_FILE="$(mkdir -p "$(dirname "$LOG_FILE")" && cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
fi

GPUS_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPUS_PER_NODE" -lt 1 ]]; then
  echo "No CUDA device is visible to torch."
  exit 1
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
SPLIT_BASE_DIR="$OUTPUT_DIR/ckpts"

mkdir -p "$OUTPUT_DIR" "$SPLIT_BASE_DIR" "$(dirname "$LOG_FILE")"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Model:  $MODEL_PATH"
echo "Data:   $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "GPUs:   $TOTAL_GPUS"
echo "LoRA:   $USE_LORA"

cd "$REPO_ROOT/src"
torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  training/train_kimi_audio_eva.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --eval_ratio "$EVAL_RATIO" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --fp16 False \
  --logging_steps 1 \
  --save_strategy no \
  --report_to tensorboard \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --gradient_checkpointing False \
  --lazy_preprocess True \
  --skip_audio_check True \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --use_lora "$USE_LORA" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_layers q_proj,k_proj,v_proj,o_proj \
  --trainable_modules "$TRAINABLE_MODULES" \
  --export_split_base_dir "$SPLIT_BASE_DIR" \
  --export_split_every_n_epochs 1 \
  --export_split_keep_last_k 5 \
  --remove_unused_columns False \
  --ddp_find_unused_parameters False
