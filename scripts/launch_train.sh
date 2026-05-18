#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Launch LoRA Training for Qwen3.5-9B on Go-Browse dataset
# 4× A100 40GB with DeepSpeed ZeRO-2
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (override via env vars) ──────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-processed_data/train}"
VAL_DATA="${VAL_DATA:-processed_data/val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3.5-9b-lora}"
NUM_GPUS="${NUM_GPUS:-4}"
RESUME="${RESUME:-false}"
REPORT_TO="${REPORT_TO:-none}"

# Hyperparams (override via env)
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"

# ── Validate ───────────────────────────────────────────────
echo "=== Go-Browse LoRA Training Launch ==="
echo "Model       : $MODEL_ID"
echo "Data (train): $TRAIN_DATA"
echo "Data (val)  : $VAL_DATA"
echo "Output      : $OUTPUT_DIR"
echo "GPUs        : $NUM_GPUS"
echo "Max seq len : $MAX_SEQ_LENGTH"
echo "Epochs      : $NUM_EPOCHS"
echo "Report to   : $REPORT_TO"
echo "Resume      : $RESUME"
echo ""

if [ ! -d "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found at '$TRAIN_DATA'"
    echo "Run: python preprocess.py first."
    exit 1
fi

# ── WandB (optional) ───────────────────────────────────────
if [ "$REPORT_TO" = "wandb" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    export WANDB_PROJECT="${WANDB_PROJECT:-go-browse}"
    echo "WandB enabled (project: $WANDB_PROJECT)"
fi

# ── Create output dirs ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Build args ─────────────────────────────────────────────
RESUME_FLAG=""
if [ "$RESUME" = "true" ]; then
    RESUME_FLAG="--resume_from_checkpoint"
fi

# ── Launch ─────────────────────────────────────────────────
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "Launching accelerate with $NUM_GPUS GPUs..."
echo "Log: $LOG_FILE"
echo ""

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes "$NUM_GPUS" \
    train_lora.py \
    --model_id "$MODEL_ID" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --report_to "$REPORT_TO" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training finished ==="
echo "Checkpoints: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"
