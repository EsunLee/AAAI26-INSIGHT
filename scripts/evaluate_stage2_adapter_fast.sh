#!/usr/bin/env bash
set -euo pipefail

# Deterministic semantic gate for a Stage-2 adapter. Defaults to 100 samples;
# set SAMPLES=876 for the complete current validation subset.

SWIFT_BIN="${SWIFT_BIN:-/data/conda_envs/insight_env/bin/swift}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
MODEL="${MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
DATASET="${DATASET:-/data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_val_stage1_history.jsonl}"
EVALUATOR="${EVALUATOR:-$PWD/CognitiveReasoning/evaluate_generated_topk_compact.py}"
SAMPLES="${SAMPLES:-100}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "${ADAPTER:-}" ]]; then
    echo "Set ADAPTER to a completed SFT checkpoint directory." >&2
    exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/EPIC-KITCHENS/stage2_sft_diagnostic_${SAMPLES}}"

for path in "$SWIFT_BIN" "$PYTHON_BIN" "$MODEL" "$DATASET" "$EVALUATOR" "$ADAPTER"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 2
    fi
done
if (( SAMPLES <= 0 || $(wc -l < "$DATASET") < SAMPLES )); then
    echo "Invalid SAMPLES=$SAMPLES for $DATASET" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
SUBSET="$OUTPUT_DIR/val_${SAMPLES}.jsonl"
RESULT="$OUTPUT_DIR/pred_adapter.jsonl"
LOG="$OUTPUT_DIR/pred_adapter.log"
METRICS="$OUTPUT_DIR/metrics.json"

if [[ -e "$RESULT" ]]; then
    echo "Refusing to overwrite existing result: $RESULT" >&2
    exit 3
fi
sed -n "1,${SAMPLES}p" "$DATASET" > "$SUBSET"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
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
        --val_dataset "$SUBSET" \
        --val_dataset_shuffle false \
        --load_data_args false \
        --max_new_tokens 256 \
        --max_batch_size 1 \
        --temperature 0.0 \
        --top_p 0.95 \
        --seed 42 \
        --result_path "$RESULT" \
        > "$LOG" 2>&1

if [[ $(wc -l < "$RESULT") -ne SAMPLES ]]; then
    echo "Inference produced an unexpected number of rows." >&2
    exit 4
fi

"$PYTHON_BIN" "$EVALUATOR" \
    --ground-truth "$SUBSET" \
    --predictions "$RESULT" \
    --horizon 20 \
    --output "$METRICS" \
    | tee "$OUTPUT_DIR/metrics.txt"

"$PYTHON_BIN" -c '
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
top = m["next_action_topk"]["top1"]
ed = m["normalized_damerau_levenshtein"]["candidate1"]
q = m["generation_quality"]
samples = m["samples"]
action_ed = ed["action"]
verb_ed = ed["verb"]
noun_ed = ed["noun"]
valid20 = 100 * q["exactly_20_valid_actions_rate"]
print("\nSEMANTIC_GATE")
print(f"samples={samples}")
print(f"top1_pct={top:.4f}")
print(f"action_ed={action_ed:.4f}")
print(f"verb_ed={verb_ed:.4f}")
print(f"noun_ed={noun_ed:.4f}")
print(f"valid20_pct={valid20:.2f}")
print("gate_pass=" + str(top > 0 or action_ed < 0.98))
' "$METRICS" | tee "$OUTPUT_DIR/summary.txt"

echo "STAGE2_ADAPTER_EVAL_COMPLETE=$OUTPUT_DIR"
