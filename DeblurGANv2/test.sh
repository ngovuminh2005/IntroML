#!/usr/bin/env bash
# Evaluate DeblurGANv2 with the same model family/checkpoint used by the notebooks.
set -euo pipefail

cd "$(dirname "$0")"

TEST_DIR="${TEST_DIR:-../datasets/deblurring/test/RealBlur_R}"
WEIGHTS_PATH="${WEIGHTS_PATH:-models/best_fpn.h5}"
OUT_DIR="${OUT_DIR:-results/realblur_r}"

python eval_folder.py \
    --test_dir "${TEST_DIR}" \
    --weights_path "${WEIGHTS_PATH}" \
    --out_dir "${OUT_DIR}"
