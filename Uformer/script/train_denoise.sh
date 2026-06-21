#!/usr/bin/env bash
set -euo pipefail

PY="python3"
ARCH="Uformer_B"
BATCH_SIZE="32"
GPU="0,1"
TRAIN_PS="128"
TRAIN_DIR="../datasets/denoising/sidd/train"
VAL_DIR="../datasets/denoising/sidd/val"
SAVE_DIR="./logs/"
ENV_NAME="_0706"
DATASET="sidd"
NEPOCH="250"
CHECKPOINT="50"
TRAIN_WORKERS="8"
EVAL_WORKERS="4"
WARMUP="true"

# Optimizer options:
#   adam                         use LR_INITIAL and WEIGHT_DECAY
#   adamw                        use LR_INITIAL and WEIGHT_DECAY
#   SingleDeviceMuonWithAuxAdam  use MUON_* for matrix-like params and MUON_AUX_* for bias/norm/scalar params
# Weight decay defaults are shared: MUON_WEIGHT_DECAY and MUON_AUX_WEIGHT_DECAY point to WEIGHT_DECAY below.
OPTIMIZER="SingleDeviceMuonWithAuxAdam"
LR_INITIAL="0.0002"
WEIGHT_DECAY="0.02"

MUON_LR="0.02"
MUON_MOMENTUM="0.95"
MUON_WEIGHT_DECAY="$WEIGHT_DECAY"
MUON_NESTEROV="true"
MUON_NS_STEPS="5"
MUON_AUX_LR="$LR_INITIAL"
MUON_AUX_WEIGHT_DECAY="$WEIGHT_DECAY"
MUON_AUX_BETA1="0.9"
MUON_AUX_BETA2="0.999"
MUON_AUX_EPS="1e-8"

MUON_ARGS=(
  --muon_lr "$MUON_LR"
  --muon_momentum "$MUON_MOMENTUM"
  --muon_weight_decay "$MUON_WEIGHT_DECAY"
  --muon_ns_steps "$MUON_NS_STEPS"
  --muon_aux_lr "$MUON_AUX_LR"
  --muon_aux_weight_decay "$MUON_AUX_WEIGHT_DECAY"
  --muon_aux_beta1 "$MUON_AUX_BETA1"
  --muon_aux_beta2 "$MUON_AUX_BETA2"
  --muon_aux_eps "$MUON_AUX_EPS"
)
case "${MUON_NESTEROV,,}" in
  true) MUON_ARGS+=(--muon_nesterov) ;;
  false) MUON_ARGS+=(--muon_no_nesterov) ;;
  *) echo "Invalid MUON_NESTEROV=$MUON_NESTEROV" >&2; exit 1 ;;
esac

EXTRA_ARGS=()
case "${WARMUP,,}" in
  true) EXTRA_ARGS+=(--warmup) ;;
  false) ;;
  *) echo "Invalid WARMUP=$WARMUP" >&2; exit 1 ;;
esac

exec "$PY" ./train/train_denoise.py \
  --arch "$ARCH" \
  --batch_size "$BATCH_SIZE" \
  --gpu "$GPU" \
  --train_ps "$TRAIN_PS" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --save_dir "$SAVE_DIR" \
  --env "$ENV_NAME" \
  --dataset "$DATASET" \
  --nepoch "$NEPOCH" \
  --checkpoint "$CHECKPOINT" \
  --train_workers "$TRAIN_WORKERS" \
  --eval_workers "$EVAL_WORKERS" \
  --optimizer "$OPTIMIZER" \
  --lr_initial "$LR_INITIAL" \
  --weight_decay "$WEIGHT_DECAY" \
  "${MUON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
