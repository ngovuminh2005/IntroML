#!/usr/bin/env bash
# Train DeblurGANv2 with the same command/config that were previously in demo_train.ipynb.
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"
AUTO_RESUME="${AUTO_RESUME:-False}"

python train.py \
    --config_path "${CONFIG_PATH}" \
    --auto_resume "${AUTO_RESUME}"
