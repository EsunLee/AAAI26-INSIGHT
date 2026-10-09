#!/usr/bin/env bash
set -euo pipefail

# Fast semantic-recovery route for the existing Stage1-predicted-history data.
# RUN_MODE=smoke runs two optimizer steps; RUN_MODE=full runs one full epoch.

SWIFT_BIN="${SWIFT_BIN:-/data/conda_envs/insight_env/bin/swift}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/EPIC-KITCHENS}"
MODEL="${MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
SFT_DATA_DIR="${SFT_DATA_DIR:-$DATA_ROOT/stage2_stage1_history_sft}"
RUN_MODE="${RUN_MODE:-full}"
GPU_IDS="${GPU_IDS:-0,1,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"

case "$RUN_MODE" in
    smoke)
        TRAIN_DATASET="${TRAIN_DATASET:-$SFT_DATA_DIR/stage2_train_sft_smoke.jsonl}"
        VAL_DATASET="${VAL_DATASET:-$SFT_DATA_DIR/stage2_val_sft_smoke.jsonl}"
        OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/stage2_sft_smoke_output}"
        EVAL_STEPS="${EVAL_STEPS:-1}"
        SAVE_STEPS="${SAVE_STEPS:-1}"
        LOGGING_STEPS="${LOGGING_STEPS:-1}"
        MAX_STEPS="${MAX_STEPS:-2}"
        SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
        ;;
    full)
        TRAIN_DATASET="${TRAIN_DATASET:-$SFT_DATA_DIR/stage2_train_sft.jsonl}"
        # A deterministic 100-row subset keeps frequent eval cheap. Full 876-row
        # validation is reserved for generation after the run passes diagnostics.
        VAL_DATASET="${VAL_DATASET:-$SFT_DATA_DIR/stage2_val_sft_100.jsonl}"
        OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/stage2_sft_fast_output}"
        EVAL_STEPS="${EVAL_STEPS:-100}"
        SAVE_STEPS="${SAVE_STEPS:-100}"
        LOGGING_STEPS="${LOGGING_STEPS:-20}"
        MAX_STEPS="${MAX_STEPS:--1}"
        # Keep every 100-step checkpoint. Action-level metrics require
        # autoregressive generation and are computed post-hoc from these models.
        SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
        ;;
    *)
        echo "RUN_MODE must be smoke or full; got: $RUN_MODE" >&2
        exit 2
        ;;
esac

for path in "$SWIFT_BIN" "$MODEL" "$TRAIN_DATASET" "$VAL_DATASET"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 2
    fi
done

IFS=',' read -r -a gpu_list <<<"$GPU_IDS"
if (( ${#gpu_list[@]} != NPROC_PER_NODE )); then
    echo "GPU_IDS contains ${#gpu_list[@]} devices but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 2
fi

if pgrep -af '/site-packages/swift/cli/(sft|rlhf)\.py' >/dev/null; then
    echo "A swift SFT/RLHF process is already running; refusing to compete for GPUs." >&2
    pgrep -af '/site-packages/swift/cli/(sft|rlhf)\.py' >&2 || true
    exit 3
fi

mkdir -p "$OUTPUT_DIR"

command=(
    sft
    --model "$MODEL"
    --model_type qwen2_5_vl
    --train_type lora
    --dataset "$TRAIN_DATASET"
    --val_dataset "$VAL_DATASET"
    --dataset_shuffle true
    --val_dataset_shuffle false
    --torch_dtype float16
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 4
    --gradient_checkpointing true
    --learning_rate 5e-5
    --lora_rank 8
    --lora_alpha 32
    --target_modules all-linear
    --eval_strategy steps
    --eval_steps "$EVAL_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --logging_steps "$LOGGING_STEPS"
    --max_length 1000
    --warmup_ratio 0.05
    --dataloader_num_workers 0
    --dataset_num_proc 1
    --deepspeed zero2
    --report_to tensorboard
    --output_dir "$OUTPUT_DIR"
)

if (( MAX_STEPS > 0 )); then
    command+=(--max_steps "$MAX_STEPS")
fi

echo "RUN_MODE=$RUN_MODE"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "VAL_DATASET=$VAL_DATASET"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GPU_IDS=$GPU_IDS NPROC_PER_NODE=$NPROC_PER_NODE"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
MAX_PIXELS=50176 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MKL_SERVICE_FORCE_INTEL=1 \
MKL_THREADING_LAYER=GNU \
    "$SWIFT_BIN" "${command[@]}"

echo "STAGE2_SFT_${RUN_MODE^^}_COMPLETE=$OUTPUT_DIR"
