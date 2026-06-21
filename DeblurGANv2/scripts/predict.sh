#!/usr/bin/env bash
# Run inference (deblurring) on one or more images with a trained model.
set -e

PROJECT_DIR="${PROJECT_DIR:-.}"
cd "$PROJECT_DIR"

# Glob/path of the blurred input image(s).
IMG_PATTERN="${IMG_PATTERN:-test_img/R_1_B.png}"
# Trained weights to load.
WEIGHTS_PATH="${WEIGHTS_PATH:-models/best_fpn.h5}"
# Where deblurred results are written.
OUT_DIR="${OUT_DIR:-submit}"

python predict.py \
    --img_pattern "$IMG_PATTERN" \
    --weights_path "$WEIGHTS_PATH" \
    --out_dir "$OUT_DIR"
