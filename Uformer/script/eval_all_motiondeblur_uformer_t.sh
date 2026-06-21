#!/usr/bin/env bash
set -euo pipefail

# Run from either IntroML root or Uformer root.
if [ -d "Uformer" ]; then
  cd Uformer
fi

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "../introml/bin/python" ]; then
  PY="../introml/bin/python"
else
  PY="python"
fi
WEIGHTS="${WEIGHTS:-./logs/motiondeblur/GoPro/Uformer_T_run1gpu/models/model_best.pth}"
GPU="${GPU:-0}"
ARCH="${ARCH:-Uformer_T}"
TRAIN_PS="${TRAIN_PS:-256}"
DATA_ROOT="../datasets/deblurring"
RESULT_ROOT="./results/deblurring"

echo "Python:  $PY"
echo "Weights: $WEIGHTS"
echo "GPU:     $GPU"
echo "Arch:    $ARCH"

mkdir -p "$DATA_ROOT/HIDE/test"
ln -sfn ../../test/HIDE/input "$DATA_ROOT/HIDE/test/input"
ln -sfn ../../test/HIDE/target "$DATA_ROOT/HIDE/test/groundtruth"

mkdir -p "$DATA_ROOT/RealBlur_J/test" "$DATA_ROOT/RealBlur_R/test"
ln -sfn ../../test/RealBlur_J/input "$DATA_ROOT/RealBlur_J/test/input"
ln -sfn ../../test/RealBlur_J/target "$DATA_ROOT/RealBlur_J/test/target"
ln -sfn ../../test/RealBlur_R/input "$DATA_ROOT/RealBlur_R/test/input"
ln -sfn ../../test/RealBlur_R/target "$DATA_ROOT/RealBlur_R/test/target"

echo "Evaluating GoPro..."
"$PY" test/test_gopro_hide.py \
  --input_dir "$DATA_ROOT/GoPro/test" \
  --result_dir "$RESULT_ROOT/GoPro/Uformer_T_run1gpu" \
  --weights "$WEIGHTS" \
  --gpus "$GPU" \
  --arch "$ARCH" \
  --train_ps "$TRAIN_PS"

echo "Evaluating HIDE..."
"$PY" test/test_gopro_hide.py \
  --input_dir "$DATA_ROOT/HIDE/test" \
  --result_dir "$RESULT_ROOT/HIDE/Uformer_T_run1gpu" \
  --weights "$WEIGHTS" \
  --gpus "$GPU" \
  --arch "$ARCH" \
  --train_ps "$TRAIN_PS"

echo "Evaluating RealBlur_J and RealBlur_R..."
"$PY" test/test_realblur.py \
  --input_dir "$DATA_ROOT" \
  --result_dir "$RESULT_ROOT" \
  --dataset RealBlur_J,RealBlur_R \
  --weights "$WEIGHTS" \
  --gpus "$GPU" \
  --arch "$ARCH" \
  --train_ps "$TRAIN_PS"

echo "Done. Results:"
echo "  $RESULT_ROOT/GoPro/Uformer_T_run1gpu/psnr_ssim.txt"
echo "  $RESULT_ROOT/HIDE/Uformer_T_run1gpu/psnr_ssim.txt"
echo "  $RESULT_ROOT/RealBlur_J/$ARCH/psnr_ssim.txt"
echo "  $RESULT_ROOT/RealBlur_R/$ARCH/psnr_ssim.txt"
