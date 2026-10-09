#!/usr/bin/env bash
set -euo pipefail

SWIFT_BIN="${SWIFT_BIN:-/data/conda_envs/insight_env/bin/swift}"
MODEL="${MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
ADAPTER="${ADAPTER:-/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773}"
DATASET="${DATASET:-/data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_val_stage1_history.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773}"
GPU_IDS="${GPU_IDS:-0,1,3}"

IFS=',' read -r -a gpu_list <<<"$GPU_IDS"
if (( ${#gpu_list[@]} == 0 )); then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

for index in 0 1 2 3 4; do
    if [[ -e "$OUTPUT_DIR/candidate_${index}.jsonl" ]]; then
        echo "Refusing to append to existing result: $OUTPUT_DIR/candidate_${index}.jsonl" >&2
        echo "Move the old result or set OUTPUT_DIR to a new directory." >&2
        exit 2
    fi
done

run_candidate() {
    local gpu="$1"
    local seed="$2"
    local temperature="$3"
    local index="$4"

    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MAX_PIXELS=50176 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MKL_SERVICE_FORCE_INTEL=1 \
    MKL_THREADING_LAYER=GNU \
    "$SWIFT_BIN" infer \
        --infer_backend pt \
        --model "$MODEL" \
        --model_type qwen2_5_vl \
        --adapters "$ADAPTER" \
        --torch_dtype float16 \
        --val_dataset "$DATASET" \
        --val_dataset_shuffle false \
        --load_data_args false \
        --max_new_tokens 256 \
        --max_batch_size 1 \
        --temperature "$temperature" \
        --top_p 0.95 \
        --seed "$seed" \
        --result_path "$OUTPUT_DIR/candidate_${index}.jsonl"
}

# Candidate 0 is deterministic and defines Top-1. Candidates 1-4 are sampled.
# Run candidates in waves so a failed physical GPU can be omitted via GPU_IDS.
pids=()
for index in 0 1 2 3 4; do
    gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
    seed=$((42 + index))
    temperature=0.9
    if (( index == 0 )); then
        temperature=0.0
    fi
    run_candidate "$gpu" "$seed" "$temperature" "$index" \
        >"$OUTPUT_DIR/candidate_${index}.log" 2>&1 &
    pids+=("$!")
    echo "Started candidate $index on physical GPU $gpu: PID $!"

    if (( ${#pids[@]} == ${#gpu_list[@]} || index == 4 )); then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done

wc -l "$OUTPUT_DIR"/candidate_*.jsonl
echo "Stage-2 candidate generation complete: $OUTPUT_DIR"
