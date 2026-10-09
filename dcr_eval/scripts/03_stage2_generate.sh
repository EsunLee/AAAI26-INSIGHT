#!/usr/bin/env bash
# 03 — Stage 2 Top-1 生成（单候选，温度 0 确定性输出）。
#
# 用法（全量三 GPU 分片）：
#   GPU=0 SPLIT_TOTAL=3 SPLIT_INDEX=0 bash scripts/03_stage2_generate.sh &
#   GPU=1 SPLIT_TOTAL=3 SPLIT_INDEX=1 bash scripts/03_stage2_generate.sh &
#   GPU=3 SPLIT_TOTAL=3 SPLIT_INDEX=2 bash scripts/03_stage2_generate.sh &
#   wait
#   分片输出会自动写回 clip_uid/action_idx；请由 run_top1_all_valid.sh 合并。
#
# 小样本（100 条 smoke）：直接 SPLIT_TOTAL=1 SPLIT_INDEX=0 跑
# 依赖：ms-swift（insight_env）、Qwen2.5-VL-7B + SFT LoRA adapter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SWIFT_BIN="${SWIFT_BIN:-/data/conda_envs/insight_env/bin/swift}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
MODEL="${MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
ADAPTER="${ADAPTER:-/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200}"
RUN_NAME="${DCR_RUN_NAME:-all_valid}"
DATASET="${DATASET:-/data/datasets/EPIC-KITCHENS/dcr_eval/runs/${RUN_NAME}/stage1_predictions/stage2_input.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/EPIC-KITCHENS/dcr_eval/runs/${RUN_NAME}/stage2_generated}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
SPLIT_TOTAL="${SPLIT_TOTAL:-1}"
GPU="${GPU:-0}"

for required in "$SWIFT_BIN" "$PYTHON_BIN" "$MODEL" "$ADAPTER" "$DATASET"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 2
    fi
done
if [[ ! -s "$DATASET" ]]; then
    echo "Dataset is empty: $DATASET" >&2
    exit 2
fi

if ! [[ "$SPLIT_INDEX" =~ ^[0-9]+$ ]] || ! [[ "$SPLIT_TOTAL" =~ ^[0-9]+$ ]]; then
    echo "SPLIT_INDEX / SPLIT_TOTAL must be non-negative integers" >&2
    exit 2
fi
if (( SPLIT_TOTAL < 1 )) || (( SPLIT_INDEX < 0 )) || (( SPLIT_INDEX >= SPLIT_TOTAL )); then
    echo "Invalid split: need 0 <= SPLIT_INDEX < SPLIT_TOTAL (got $SPLIT_INDEX / $SPLIT_TOTAL)" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

# 分片：按行均分输入，当前分片写到 /tmp
if [[ "$SPLIT_TOTAL" -gt 1 ]]; then
    SHARD_TMP_DIR="$(mktemp -d /tmp/dcr-stage2-shard.XXXXXX)"
    SHARD_IN="${SHARD_TMP_DIR}/input.jsonl"
    trap 'rm -rf "$SHARD_TMP_DIR"' EXIT
    "$PYTHON_BIN" - "$DATASET" "$SPLIT_INDEX" "$SPLIT_TOTAL" "$SHARD_IN" <<'EOF'
import sys
src, idx, total, dst = sys.argv[1:]
lines = [ln for ln in open(src, encoding="utf-8") if ln.strip()]
n = len(lines)
lo = n * int(idx) // int(total)
hi = n * (int(idx) + 1) // int(total)
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(lines[lo:hi])
print(f"shard {idx}/{total}: lines {lo}-{hi} of {n} → {dst}")
EOF
else
    SHARD_IN="$DATASET"
fi

OUT="$OUTPUT_DIR/candidate_0_part_${SPLIT_INDEX}.jsonl"
if [[ -e "$OUT" ]]; then
    expected_lines="$(grep -cve '^[[:space:]]*$' "$SHARD_IN")"
    actual_lines="$(grep -cve '^[[:space:]]*$' "$OUT" || true)"
    if [[ "$actual_lines" -eq "$expected_lines" && "$expected_lines" -gt 0 ]] && \
       "$PYTHON_BIN" "$SCRIPT_DIR/validate_generated_jsonl.py" "$OUT" "$expected_lines"; then
        # 兼容旧缓存：原子地补写并核对稳定样本 ID；已有 ID 冲突时直接失败。
        "$PYTHON_BIN" "$SCRIPT_DIR/attach_prediction_ids.py" \
            --input "$SHARD_IN" \
            --predictions "$OUT" \
            --output "$OUT"
        "$PYTHON_BIN" "$SCRIPT_DIR/validate_generated_jsonl.py" \
            "$OUT" "$expected_lines" --require-ids
        echo "Reusing complete Stage2 shard: $OUT ($actual_lines lines)"
        exit 0
    fi
    backup="${OUT}.incomplete.$(date +%Y%m%d-%H%M%S).$$"
    mv "$OUT" "$backup"
    echo "Preserved incomplete shard ($actual_lines/$expected_lines) as: $backup" >&2
fi

RAW_OUT="${OUT}.swift_tmp.$$"
trap 'rm -rf "${SHARD_TMP_DIR:-}"; rm -f "$RAW_OUT"' EXIT

CUDA_VISIBLE_DEVICES="$GPU" \
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
    --val_dataset "$SHARD_IN" \
    --val_dataset_shuffle false \
    --load_data_args false \
    --max_new_tokens 256 \
    --max_batch_size 1 \
    --temperature 0.0 \
    --top_p 0.95 \
    --seed 42 \
    --result_path "$RAW_OUT"

wc -l "$RAW_OUT"
expected_lines="$(grep -cve '^[[:space:]]*$' "$SHARD_IN")"
actual_lines="$(grep -cve '^[[:space:]]*$' "$RAW_OUT")"
if [[ "$actual_lines" -ne "$expected_lines" ]]; then
    echo "Generated shard line count mismatch: output=$actual_lines input=$expected_lines" >&2
    exit 5
fi
"$PYTHON_BIN" "$SCRIPT_DIR/validate_generated_jsonl.py" "$RAW_OUT" "$expected_lines"
"$PYTHON_BIN" "$SCRIPT_DIR/attach_prediction_ids.py" \
    --input "$SHARD_IN" \
    --predictions "$RAW_OUT" \
    --output "$OUT"
"$PYTHON_BIN" "$SCRIPT_DIR/validate_generated_jsonl.py" \
    "$OUT" "$expected_lines" --require-ids
echo "Stage-2 Top-1 generation complete (split $SPLIT_INDEX/$SPLIT_TOTAL): $OUT"
