#!/usr/bin/env bash
set -euo pipefail

# Batch-run 10-fold training and prediction for ISRUC.
# Usage: nohup bash script/run_cv_isruc.sh &

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT_DIR/config/isruc.py"
OUT_TRAIN="$ROOT_DIR/output/isruc_train"
OUT_PRED="$ROOT_DIR/output/isruc_predict"
LOG_DIR="$ROOT_DIR/output/logs"
GPU=0

mkdir -p "$OUT_TRAIN" "$OUT_PRED" "$LOG_DIR"

for FOLD in $(seq 0 9); do
  echo "=== Fold ${FOLD} ==="
  FOLD_LOG="$LOG_DIR/train_fold_${FOLD}.log"
  # Train
  python3 "$ROOT_DIR/train.py" \
    --config_file "$CONFIG" \
    --fold_idx ${FOLD} \
    --output_dir "$OUT_TRAIN" \
    --restart \
    --log_file "$FOLD_LOG" \
    --gpu ${GPU}

  # Predict on test set using best checkpoint
  PRED_OUT="$OUT_PRED/fold_${FOLD}"
  mkdir -p "$PRED_OUT"
  PRED_LOG="$LOG_DIR/predict_fold_${FOLD}.log"
  python3 "$ROOT_DIR/predict.py" \
    --config_file "$CONFIG" \
    --model_dir "$OUT_TRAIN" \
    --output_dir "$PRED_OUT" \
    --log_file "$PRED_LOG" \
    --use-best \
    --gpu ${GPU} \
    --fold_idx ${FOLD}

  echo "Fold ${FOLD} finished. Outputs: ${PRED_OUT}"
done

echo "All folds finished. Predictions saved under $OUT_PRED"
