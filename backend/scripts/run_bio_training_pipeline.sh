#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
DATA_DIR="${1:-$ROOT_DIR/backend/data/bio_continuous}"
ARTIFACTS_DIR="${2:-$ROOT_DIR/backend/artifacts}"

echo "[bio_pipeline] root=$ROOT_DIR"
echo "[bio_pipeline] data_dir=$DATA_DIR"
echo "[bio_pipeline] artifacts_dir=$ARTIFACTS_DIR"

"$PY_BIN" "$ROOT_DIR/backend/train/train_pose_bio_segmenter.py" \
  --data-dir "$DATA_DIR" \
  --save-dir "$ARTIFACTS_DIR/bio_training" \
  --thresholds-out "$ARTIFACTS_DIR/bio_thresholds.json" \
  --epochs 20 \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout 0.2 \
  --bidirectional true \
  --alpha-phrase 1.0 \
  --threshold-grid "0.3,0.4,0.5,0.6,0.7" \
  --boundary-tolerance 0 \
  --segment-iou-threshold 0.5

"$PY_BIN" "$ROOT_DIR/backend/train/export_bio_segmenter_onnx.py" \
  --checkpoint "$ARTIFACTS_DIR/bio_training/best_model.pt" \
  --onnx-out "$ARTIFACTS_DIR/bio_segmenter.onnx" \
  --thresholds-in "$ARTIFACTS_DIR/bio_thresholds.json" \
  --thresholds-out "$ARTIFACTS_DIR/bio_thresholds.json" \
  --bio-config-out "$ARTIFACTS_DIR/bio_config.json" \
  --window-size 256 \
  --dynamic-time true

echo "[bio_pipeline] done"
