#!/usr/bin/env bash
set -euo pipefail

# EK55 long-term future-verb mAP under the EGO-TOPO/INSIGHT protocol.
# This is a feature-limited baseline: it reuses the available Stage-1 RGB
# action features, while labels/splits/observation ratios follow EGO-TOPO.
REPO_ROOT="${REPO_ROOT:-/home/amax/ldy/git/AAAI26-INSIGHT}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/EPIC-KITCHENS}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/egotopo_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ek55_future_verb_map_output}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-${DATA_ROOT}/annotations/EPIC_train_action_labels.csv}"
MANY_SHOT_CSV="${MANY_SHOT_CSV:-${DATA_ROOT}/annotations/EPIC_many_shot_verbs.csv}"
FRAME_FEATURES="${FRAME_FEATURES:-${DATA_ROOT}/features/frame_features}"
SPLIT_URL="${SPLIT_URL:-https://dl.fbaipublicfiles.com/ego-topo/data/epic/splits.zip}"
GPU="${GPU:-0}"

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "$SPLIT_ROOT/split/train_S1.csv" || ! -f "$SPLIT_ROOT/split/val_S1.csv" ]]; then
  mkdir -p "$SPLIT_ROOT"
  split_zip="/tmp/egotopo_epic_splits.zip"
  curl -L "$SPLIT_URL" -o "$split_zip"
  unzip -oq "$split_zip" -d "$SPLIT_ROOT"
fi

mkdir -p "$OUTPUT_DIR"
many_shot_args=()
if [[ -f "$MANY_SHOT_CSV" ]]; then
  many_shot_args=(--many-shot-verbs "$MANY_SHOT_CSV")
else
  echo "WARNING: $MANY_SHOT_CSV not found; deriving >100 classes from the full EK55 annotation CSV" >&2
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  evaluation/train_ek55_future_verb_map.py \
  --split-dir "$SPLIT_ROOT/split" \
  --annotations-csv "$ANNOTATIONS_CSV" \
  --frame-features "$FRAME_FEATURES" \
  "${many_shot_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --input-dim 1024 \
  --hidden-dim 512 \
  --epochs 60 \
  --patience 8 \
  --batch-size 64 \
  --num-workers 0 \
  --learning-rate 1e-3 \
  | tee "$OUTPUT_DIR/train_and_metrics.log"
