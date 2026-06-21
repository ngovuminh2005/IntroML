#!/usr/bin/env bash
# Evaluate/infer DeblurGAN with the same checkpoint settings from demo.ipynb.
set -euo pipefail

cd "$(dirname "$0")"

NAME="${NAME:-gopro}"
WHICH_EPOCH="${WHICH_EPOCH:-200}"
BLURRED_DIR="${BLURRED_DIR:-../datasets/deblurring/test/GoPro/input}"
SHARP_DIR="${SHARP_DIR:-../datasets/deblurring/test/GoPro/target}"
RESULTS_DIR="${RESULTS_DIR:-results/gopro}"
GPU_IDS="${GPU_IDS:-0}"

python deblur_eval.py \
    --name "${NAME}" \
    --which_epoch "${WHICH_EPOCH}" \
    --blurred_dir "${BLURRED_DIR}" \
    --sharp_dir "${SHARP_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --gpu_ids "${GPU_IDS}"
