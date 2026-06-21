#!/usr/bin/env bash
# Evaluate a trained model over a test folder (computes metrics, e.g. PSNR/SSIM).
# The test folder must contain matching blurred / sharp image pairs.
set -e

PROJECT_DIR="${PROJECT_DIR:-.}"
cd "$PROJECT_DIR"

# Folder of test image pairs to evaluate on.
TEST_DIR="${TEST_DIR:-datasets/deblurring/test/RealBlur_R}"
# Trained weights to evaluate.
WEIGHTS_PATH="${WEIGHTS_PATH:-models/best_fpn.h5}"

python eval_folder.py \
    --test_dir "$TEST_DIR" \
    --weights_path "$WEIGHTS_PATH"
