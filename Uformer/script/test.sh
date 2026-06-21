#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python3}"
DATASET="${DATASET:-GoPro}"
ARCH="${ARCH:-Uformer_B}"
GPUS="${GPUS:-0}"
WEIGHTS="${WEIGHTS:-./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth}"
INPUT_DIR="${INPUT_DIR:-../datasets/deblurring}"
RESULT_DIR="${RESULT_DIR:-./results/deblurring}"

case "$DATASET" in
  GoPro)
    "$PY" test/test_gopro_hide.py \
      --input_dir "${INPUT_DIR}/GoPro/test/" \
      --result_dir "${RESULT_DIR}/GoPro/${ARCH}/" \
      --weights "$WEIGHTS" \
      --gpus "$GPUS" \
      --arch "$ARCH"
    ;;
  HIDE)
    "$PY" test/test_gopro_hide.py \
      --input_dir "${INPUT_DIR}/HIDE/test/" \
      --result_dir "${RESULT_DIR}/HIDE/${ARCH}/" \
      --weights "$WEIGHTS" \
      --gpus "$GPUS" \
      --arch "$ARCH"
    ;;
  RealBlur_J|RealBlur_R|RealBlur_J,RealBlur_R)
    "$PY" test/test_realblur.py \
      --input_dir "$INPUT_DIR" \
      --result_dir "$RESULT_DIR" \
      --dataset "$DATASET" \
      --weights "$WEIGHTS" \
      --gpus "$GPUS" \
      --arch "$ARCH"
    ;;
  *)
    echo "Unsupported DATASET=$DATASET. Use GoPro, HIDE, RealBlur_J, RealBlur_R, or RealBlur_J,RealBlur_R." >&2
    exit 1
    ;;
esac
