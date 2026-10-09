#!/usr/bin/env bash
set -euo pipefail

# Official-style EK55 one-second action anticipation pipeline.
# MODE: prepare | probe | extract | train | all
MODE="${MODE:-all}"
REPO_ROOT="${REPO_ROOT:-/home/amax/ldy/git/AAAI26-INSIGHT}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/EPIC-KITCHENS}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/ek55_anticipation_official}"
FEATURE_DIR="${FEATURE_DIR:-${DATA_ROOT}/features/anticipation_rgb_tau1}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ek55_anticipation_topk_output}"
EGO_CHECKPOINT="${EGO_CHECKPOINT:-/data/pretrain_model/EgoVideo/checkpoints/large_best.pt}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-${DATA_ROOT}/annotations/EPIC_train_action_labels.csv}"
ALLOW_FLAT="${ALLOW_FLAT:-0}"
FRAME_TEMPLATE="${FRAME_TEMPLATE:-}"

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

prepare() {
  "$PYTHON_BIN" evaluation/prepare_ek55_anticipation.py \
    --annotations-csv "$ANNOTATIONS_CSV" \
    --output-dir "$MANIFEST_DIR" \
    --fps 60 \
    --time-step 0.25 \
    --sequence-length 14 \
    --anticipation-time 1.0
}

extract_args=(
  --manifest-dir "$MANIFEST_DIR"
  --data-root "$DATA_ROOT"
  --checkpoint "$EGO_CHECKPOINT"
  --output-dir "$FEATURE_DIR"
  --batch-actions 2
  --nearest-radius 15
)
if [[ -n "$FRAME_TEMPLATE" ]]; then
  extract_args+=(--frame-template "$FRAME_TEMPLATE")
fi
if [[ "$ALLOW_FLAT" == "1" ]]; then
  extract_args+=(--allow-ambiguous-flat-layout)
fi

probe() {
  "$PYTHON_BIN" feature_extraction/extract_ek55_anticipation_rgb.py \
    "${extract_args[@]}" --probe-only
}

extract() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    feature_extraction/extract_ek55_anticipation_rgb.py \
    "${extract_args[@]}" --splits train \
    > /tmp/ek55_anticipation_extract_train.log 2>&1 &
  train_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" \
    feature_extraction/extract_ek55_anticipation_rgb.py \
    "${extract_args[@]}" --splits val \
    > /tmp/ek55_anticipation_extract_val.log 2>&1 &
  val_pid=$!
  echo "Extraction PIDs: train=${train_pid} val=${val_pid}"
  wait "$train_pid"
  wait "$val_pid"
}

train_topk() {
  mkdir -p "$OUTPUT_DIR"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" HandObject/train_anticipation_topk.py \
    --manifest-dir "$MANIFEST_DIR" \
    --features-dir "$FEATURE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --input-dim 1024 \
    --epochs 40 \
    --patience 6 \
    --batch-size 64 \
    --num-workers 0 \
    --learning-rate 8e-5 \
    | tee "$OUTPUT_DIR/train_and_metrics.log"
}

case "$MODE" in
  prepare) prepare ;;
  probe) prepare; probe ;;
  extract) extract ;;
  train) train_topk ;;
  all) prepare; probe; extract; train_topk ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac
