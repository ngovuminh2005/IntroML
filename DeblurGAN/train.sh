#!/usr/bin/env bash
# Train DeblurGAN with the same settings that were previously in demo.ipynb.
set -euo pipefail

cd "$(dirname "$0")"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gopro}"
export DATAROOT="${DATAROOT:-../datasets/deblurring/train/GoPro}"
export PHASE="${PHASE:-full}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CROP_SIZE="${CROP_SIZE:-256}"

python train.py \
    --name "${EXPERIMENT_NAME}" \
    --dataroot "${DATAROOT}" \
    --phase "${PHASE}" \
    --dataset_mode paired \
    --learn_residual \
    --resize_or_crop crop \
    --fineSize "${CROP_SIZE}" \
    --gan_type wgan-gp \
    --batchSize "${BATCH_SIZE}" \
    --nThreads "${NUM_WORKERS}" \
    --display_id 0 \
    --save_latest_freq 100 \
    --print_freq 20
