#!/usr/bin/env bash
# 一键跑通 INSIGHT Stage1 → Stage2 → 生成式 Top-1（DCR-valid proxy）。
#
# 快速验证：RUN_MODE=smoke bash scripts/run_top1_all_valid.sh
# 全量评测：RUN_MODE=full  bash scripts/run_top1_all_valid.sh
# 可覆盖 GPU：STAGE1_GPU=0 STAGE2_GPUS=0,1,3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
RUN_MODE="${RUN_MODE:-full}"
STAGE1_GPU="${STAGE1_GPU:-0}"
STAGE2_GPUS="${STAGE2_GPUS:-0,1,3}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-100}"
STAGE2_MODEL="${STAGE2_MODEL:-/data/pretrain_model/Qwen2.5-VL-7B-Instruct}"
STAGE2_ADAPTER="${STAGE2_ADAPTER:-/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200}"
export DCR_RUN_NAME="${DCR_RUN_NAME:-all_valid}"

for required in "$PYTHON_BIN" scripts/00_audit.py scripts/01_prepare_features.py \
                scripts/02_stage1_infer.py scripts/03_stage2_generate.sh scripts/04_evaluate.py \
                scripts/attach_prediction_ids.py scripts/validate_generated_jsonl.py; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required executable/script: $required" >&2
        exit 2
    fi
done

if [[ "$RUN_MODE" != "smoke" && "$RUN_MODE" != "full" ]]; then
    echo "RUN_MODE must be smoke or full (got: $RUN_MODE)" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$STAGE2_GPUS"
SPLIT_TOTAL="${#GPU_IDS[@]}"
if (( SPLIT_TOTAL < 1 )); then
    echo "STAGE2_GPUS is empty" >&2
    exit 2
fi
for gpu in "$STAGE1_GPU" "${GPU_IDS[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "GPU identifiers must be physical numeric indices, got: $gpu" >&2
        exit 2
    fi
done

echo "[environment] 检查 Python 依赖与 ${SPLIT_TOTAL} 张 Stage2 GPU 的 CUDA 可用性"
CUDA_VISIBLE_DEVICES="$STAGE2_GPUS" "$PYTHON_BIN" - "$SPLIT_TOTAL" <<'PY'
import sys
import cv2
import numpy
import pandas
import torch
import transformers
import tqdm

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"CUDA device count mismatch: expected={expected}, visible={actual}")
for index in range(actual):
    torch.zeros(1, device=f"cuda:{index}")
    print(f"  cuda:{index} {torch.cuda.get_device_name(index)} PASS", flush=True)
PY

RUN_ROOT="/data/datasets/EPIC-KITCHENS/dcr_eval/runs/${DCR_RUN_NAME}"
FULL_INPUT="${RUN_ROOT}/stage1_predictions/stage2_input.jsonl"
if [[ "$RUN_MODE" == "smoke" ]]; then
    DATASET="${RUN_ROOT}/stage1_predictions/stage2_input_smoke.jsonl"
    OUTPUT_DIR="${RUN_ROOT}/stage2_generated_smoke"
    METRICS="${RUN_ROOT}/evaluation/metrics_smoke.json"
else
    DATASET="$FULL_INPUT"
    OUTPUT_DIR="${RUN_ROOT}/stage2_generated"
    METRICS="${RUN_ROOT}/evaluation/metrics.json"
fi

echo "[phase 0/4] CPU 审计：确认数据、类别映射与训练集重叠情况"
"$PYTHON_BIN" scripts/00_audit.py

echo "[phase 1/4] 提取 Stage1 所需的 1024 维 RGB 特征（已存在的 .pt 会跳过）"
CUDA_VISIBLE_DEVICES="$STAGE1_GPU" "$PYTHON_BIN" scripts/01_prepare_features.py

echo "[phase 2/4] Stage1 Top-1 识别，并构造 8 段历史 + 20 段完整未来的 Stage2 输入"
CUDA_VISIBLE_DEVICES="$STAGE1_GPU" "$PYTHON_BIN" scripts/02_stage1_infer.py

if [[ ! -s "$FULL_INPUT" ]]; then
    echo "Stage2 input is missing or empty: $FULL_INPUT" >&2
    exit 3
fi

if [[ "$RUN_MODE" == "smoke" ]]; then
    head -n "$SMOKE_SAMPLES" "$FULL_INPUT" > "$DATASET"
    echo "[smoke] 只评测前 $(wc -l < "$DATASET") 条：$DATASET"
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$METRICS")"
MERGED="${OUTPUT_DIR}/candidate_0.jsonl"
input_lines="$(grep -cve '^[[:space:]]*$' "$DATASET")"
dataset_sha256="$(sha256sum "$DATASET" | awk '{print $1}')"
GENERATION_SPEC="${OUTPUT_DIR}/generation_spec.txt"
current_spec="$(printf 'dataset_sha256=%s\nmodel=%s\nadapter=%s\nsplit_total=%s\nid_alignment=clip_uid_action_idx_v1\n' \
    "$dataset_sha256" "$STAGE2_MODEL" "$STAGE2_ADAPTER" "$SPLIT_TOTAL")"
if [[ -e "$GENERATION_SPEC" ]]; then
    saved_spec="$(cat "$GENERATION_SPEC")"
    if [[ "$saved_spec" != "$current_spec" ]]; then
        echo "Stage2 generation cache spec mismatch: $GENERATION_SPEC" >&2
        echo "Choose a new DCR_RUN_NAME; do not reuse predictions from different input/model settings." >&2
        exit 2
    fi
elif compgen -G "${OUTPUT_DIR}/candidate_0*.jsonl" > /dev/null; then
    echo "Stage2 outputs exist but generation_spec.txt is missing: $OUTPUT_DIR" >&2
    echo "Choose a new DCR_RUN_NAME so cache provenance is unambiguous." >&2
    exit 2
else
    printf '%s\n' "$current_spec" > "$GENERATION_SPEC"
fi

reuse_merged=0
if [[ -e "$MERGED" ]]; then
    merged_lines="$(grep -cve '^[[:space:]]*$' "$MERGED" || true)"
    if [[ "$merged_lines" -eq "$input_lines" && "$input_lines" -gt 0 ]] && \
       "$PYTHON_BIN" scripts/validate_generated_jsonl.py \
           "$MERGED" "$input_lines" --require-ids; then
        echo "[resume] 复用已完整合并的预测: $MERGED ($merged_lines lines)"
        reuse_merged=1
    else
        backup="${MERGED}.invalid.$(date +%Y%m%d-%H%M%S).$$"
        mv "$MERGED" "$backup"
        echo "[resume] 保留不完整合并文件 ($merged_lines/$input_lines): $backup"
    fi
fi

if (( reuse_merged == 0 )); then
    echo "[phase 3/4] Stage2 温度 0 Top-1 生成：${SPLIT_TOTAL} 个分片，物理 GPU=${STAGE2_GPUS}"
    pids=()
    for ((idx = 0; idx < SPLIT_TOTAL; idx++)); do
        GPU="${GPU_IDS[$idx]}" \
        SPLIT_TOTAL="$SPLIT_TOTAL" \
        SPLIT_INDEX="$idx" \
        DATASET="$DATASET" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        MODEL="$STAGE2_MODEL" \
        ADAPTER="$STAGE2_ADAPTER" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash scripts/03_stage2_generate.sh \
            > "${OUTPUT_DIR}/shard_${idx}.log" 2>&1 &
        pids+=("$!")
        echo "  shard=$idx gpu=${GPU_IDS[$idx]} pid=${pids[$idx]}"
    done

    failed=0
    for ((idx = 0; idx < SPLIT_TOTAL; idx++)); do
        if ! wait "${pids[$idx]}"; then
            echo "Stage2 shard $idx failed; inspect ${OUTPUT_DIR}/shard_${idx}.log" >&2
            failed=1
        fi
    done
    if (( failed != 0 )); then
        exit 4
    fi

    MERGED_TMP="${MERGED}.merge_tmp.$$"
    : > "$MERGED_TMP"
    trap 'rm -f "${MERGED_TMP:-}"' EXIT
    for ((idx = 0; idx < SPLIT_TOTAL; idx++)); do
        part="${OUTPUT_DIR}/candidate_0_part_${idx}.jsonl"
        [[ -s "$part" ]] || { echo "Missing/empty shard: $part" >&2; exit 4; }
        "$PYTHON_BIN" scripts/validate_generated_jsonl.py \
            "$part" "$(grep -cve '^[[:space:]]*$' "$part")" --require-ids
        cat "$part" >> "$MERGED_TMP"
    done
    "$PYTHON_BIN" scripts/validate_generated_jsonl.py \
        "$MERGED_TMP" "$input_lines" --require-ids
    mv "$MERGED_TMP" "$MERGED"
    trap - EXIT
fi

output_lines="$(grep -cve '^[[:space:]]*$' "$MERGED")"
echo "[merge] input=$input_lines output=$output_lines"
if [[ "$input_lines" -ne "$output_lines" ]]; then
    echo "Merged prediction count does not match input count; evaluation aborted." >&2
    exit 5
fi
"$PYTHON_BIN" scripts/validate_generated_jsonl.py \
    "$MERGED" "$input_lines" --require-ids

echo "[phase 4/4] 精确匹配 GT，计算唯一的最终指标 Top-1"
"$PYTHON_BIN" scripts/04_evaluate.py \
    --input "$DATASET" \
    --predictions-dir "$OUTPUT_DIR" \
    --output "$METRICS"

echo "TOP1_PIPELINE_COMPLETE=$METRICS"
cat "$METRICS"
