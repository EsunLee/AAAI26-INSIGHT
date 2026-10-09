#!/usr/bin/env bash
set -euo pipefail

SWIFT_BIN="${SWIFT_BIN:-/data/conda_envs/insight_env/bin/swift}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
MODEL="${MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
ADAPTER="${ADAPTER:-/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773}"
GT_DATASET="${GT_DATASET:-/data/datasets/EPIC-KITCHENS/stage2_val_lta_context.jsonl}"
PRED_DATASET="${PRED_DATASET:-/data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_val_stage1_history.jsonl}"
EVALUATOR="${EVALUATOR:-$PWD/CognitiveReasoning/evaluate_generated_topk_compact.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/EPIC-KITCHENS/stage2_diagnostic_100_checkpoint1773}"
SAMPLES="${SAMPLES:-100}"

for path in "$GT_DATASET" "$PRED_DATASET" "$EVALUATOR"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
done
if [[ ! -d "$ADAPTER" ]]; then
    echo "Missing adapter directory: $ADAPTER" >&2
    exit 2
fi
if (( $(wc -l < "$GT_DATASET") < SAMPLES || $(wc -l < "$PRED_DATASET") < SAMPLES )); then
    echo "A diagnostic source dataset has fewer than $SAMPLES rows." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
GT_SUBSET="$OUTPUT_DIR/gt_history_${SAMPLES}.jsonl"
PRED_SUBSET="$OUTPUT_DIR/pred_history_${SAMPLES}.jsonl"
sed -n "1,${SAMPLES}p" "$GT_DATASET" > "$GT_SUBSET"
sed -n "1,${SAMPLES}p" "$PRED_DATASET" > "$PRED_SUBSET"

labels=(gt_base gt_adapter pred_base pred_adapter)
for label in "${labels[@]}"; do
    if [[ -e "$OUTPUT_DIR/${label}.jsonl" ]]; then
        echo "Refusing to overwrite existing result: $OUTPUT_DIR/${label}.jsonl" >&2
        exit 2
    fi
done

run_one() {
    local gpu="$1"
    local label="$2"
    local dataset="$3"
    local use_adapter="$4"
    local -a command=(
        infer
        --infer_backend pt
        --model "$MODEL"
        --model_type qwen2_5_vl
        --torch_dtype float16
        --val_dataset "$dataset"
        --val_dataset_shuffle false
        --load_data_args false
        --max_new_tokens 256
        --max_batch_size 1
        --temperature 0.0
        --top_p 0.95
        --seed 42
        --result_path "$OUTPUT_DIR/${label}.jsonl"
    )
    if [[ "$use_adapter" == true ]]; then
        command+=(--adapters "$ADAPTER")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MAX_PIXELS=50176 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MKL_SERVICE_FORCE_INTEL=1 \
    MKL_THREADING_LAYER=GNU \
        "$SWIFT_BIN" "${command[@]}" \
        > "$OUTPUT_DIR/${label}.log" 2>&1
}

# Wave 1: three independent comparisons on the three healthy physical GPUs.
run_one 0 gt_base "$GT_SUBSET" false & p0=$!
run_one 1 gt_adapter "$GT_SUBSET" true & p1=$!
run_one 3 pred_base "$PRED_SUBSET" false & p2=$!
echo "Wave 1 PIDs: gt_base=$p0 gt_adapter=$p1 pred_base=$p2"
wait "$p0"
wait "$p1"
wait "$p2"

# Wave 2: the fourth comparison.
run_one 0 pred_adapter "$PRED_SUBSET" true & p3=$!
echo "Wave 2 PID: pred_adapter=$p3"
wait "$p3"

for label in "${labels[@]}"; do
    if [[ $(wc -l < "$OUTPUT_DIR/${label}.jsonl") -ne SAMPLES ]]; then
        echo "Unexpected row count for $label" >&2
        exit 3
    fi
    dataset="$PRED_SUBSET"
    if [[ "$label" == gt_* ]]; then
        dataset="$GT_SUBSET"
    fi
    "$PYTHON_BIN" "$EVALUATOR" \
        --ground-truth "$dataset" \
        --predictions "$OUTPUT_DIR/${label}.jsonl" \
        --horizon 20 \
        --output "$OUTPUT_DIR/${label}_metrics.json" \
        > "$OUTPUT_DIR/${label}_metrics.txt"
done

"$PYTHON_BIN" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print("condition\ttop1_pct\taction_ed\tverb_ed\tnoun_ed\tvalid20_pct")
for label in ("gt_base", "gt_adapter", "pred_base", "pred_adapter"):
    data = json.loads((root / f"{label}_metrics.json").read_text())
    top = data["next_action_topk"]["top1"]
    ed = data["normalized_damerau_levenshtein"]["candidate1"]
    valid = 100 * data["generation_quality"]["exactly_20_valid_actions_rate"]
    print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}".format(
        label, top, ed["action"], ed["verb"], ed["noun"], valid
    ))
' "$OUTPUT_DIR" | tee "$OUTPUT_DIR/summary.tsv"

echo "DIAGNOSTIC_COMPLETE=$OUTPUT_DIR"
