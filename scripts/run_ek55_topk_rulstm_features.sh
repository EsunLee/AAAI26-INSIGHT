#!/usr/bin/env bash
set -euo pipefail

# Fast, leakage-free EK55 Top-1/Top-5 path using RULSTM's official TSN-RGB LMDB.
REPO_ROOT="${REPO_ROOT:-/home/amax/ldy/git/AAAI26-INSIGHT}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/EPIC-KITCHENS}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/insight_env/bin/python3}"
RULSTM_ROOT="${RULSTM_ROOT:-${DATA_ROOT}/rulstm_source}"
LMDB_DIR="${LMDB_DIR:-${DATA_ROOT}/rulstm_ek55_rgb}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/ek55_anticipation_rulstm_rgb}"
FEATURE_DIR="${FEATURE_DIR:-${DATA_ROOT}/features/anticipation_rulstm_rgb_tau1}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ek55_topk_rulstm_rgb_output}"
RULSTM_ARCHIVE_URL="${RULSTM_ARCHIVE_URL:-https://github.com/fpv-iplab/rulstm/archive/refs/heads/master.tar.gz}"
RGB_LMDB_URL="${RGB_LMDB_URL:-https://iplab.dmi.unict.it/sharing/rulstm/features/rgb/data.mdb}"

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if ! "$PYTHON_BIN" -c 'import lmdb' >/dev/null 2>&1; then
  echo "Missing lmdb. Install once with:" >&2
  echo "  $PYTHON_BIN -m pip install lmdb" >&2
  exit 2
fi

if [[ ! -f "$RULSTM_ROOT/RULSTM/data/ek55/training.csv" ]]; then
  mkdir -p "$RULSTM_ROOT"
  archive="/tmp/rulstm-master.tar.gz"
  curl -L "$RULSTM_ARCHIVE_URL" -o "$archive"
  tar -xzf "$archive" --strip-components=1 -C "$RULSTM_ROOT"
fi

mkdir -p "$LMDB_DIR"
if [[ ! -f "$LMDB_DIR/data.mdb" ]]; then
  echo "Downloading official RGB LMDB (6,515,089,408 bytes)..."
  curl -L --continue-at - "$RGB_LMDB_URL" -o "$LMDB_DIR/data.mdb"
fi

"$PYTHON_BIN" feature_extraction/export_rulstm_lmdb_sequences.py \
  --rulstm-data-dir "$RULSTM_ROOT/RULSTM/data/ek55" \
  --lmdb-dir "$LMDB_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --features-dir "$FEATURE_DIR" \
  --fps 30 \
  --time-step 0.25 \
  --anticipation-time 1.0 \
  --earliest-time 3.5 \
  --input-dim 1024

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

