#!/usr/bin/env bash
set -euo pipefail

if [ -d "Restormer" ]; then
  cd Restormer
fi
ROOT="$(pwd)"
MOTION_DIR="$ROOT/Motion_Deblurring"

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/../introml/bin/python" ]; then
  PY="$ROOT/../introml/bin/python"
else
  PY="python"
fi

WEIGHTS="${WEIGHTS:-$ROOT/experiments/Deblurring_Restormer_dim48_fresh/models/net_g_latest.pth}"
DATA_ROOT="${DATA_ROOT:-$ROOT/../datasets/deblurring}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/deblurring/Deblurring_Restormer_dim48_fresh}"

if [ ! -f "$WEIGHTS" ]; then
  echo "Weights not found: $WEIGHTS" >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Dataset root not found: $DATA_ROOT" >&2
  exit 1
fi

export DATA_ROOT RESULT_ROOT

cd "$MOTION_DIR"
mkdir -p "$RESULT_ROOT"

for DATASET in GoPro HIDE RealBlur_J RealBlur_R; do
  echo "Evaluating $DATASET..."
  "$PY" test.py \
    --input_dir "$DATA_ROOT" \
    --result_dir "$RESULT_ROOT" \
    --weights "$WEIGHTS" \
    --dataset "$DATASET"
done

echo "Computing metrics..."
"$PY" - <<'PY'
import concurrent.futures
import math
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from skimage import io
from skimage.metrics import structural_similarity

DATA_ROOT = Path(os.environ['DATA_ROOT'])
RESULT_ROOT = Path(os.environ['RESULT_ROOT'])


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim_channel(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    return float(np.mean([_ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))


def image_align(deblurred, gt):
    z = deblurred
    x = gt
    zs = (np.sum(x * z) / np.sum(z * z)) * z
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0)
    _, warp_matrix = cv2.findTransformECC(
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
        warp_matrix,
        cv2.MOTION_HOMOGRAPHY,
        criteria,
        inputMask=None,
        gaussFiltSize=5,
    )
    target_shape = x.shape
    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return zr * cr, x * cr, cr


def compute_psnr_masked(image_true, image_test, image_mask):
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10(1.0 / err)


def compute_ssim_masked(tar_img, prd_img, cr1):
    _, ssim_map = structural_similarity(
        tar_img,
        prd_img,
        channel_axis=-1,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=1.0,
        full=True,
    )
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    return float(np.mean(ssim))


def eval_pair_gopro_hide(args):
    gt_path, pred_path = args
    gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(cv2.imread(str(pred_path)), cv2.COLOR_BGR2RGB)
    return calculate_psnr(gt, pred), calculate_ssim(gt, pred)


def eval_pair_realblur(args):
    gt_path, pred_path = args
    tar_img = io.imread(gt_path).astype(np.float32) / 255.0
    prd_img = io.imread(pred_path).astype(np.float32) / 255.0
    prd_img, tar_img, cr1 = image_align(prd_img, tar_img)
    return compute_psnr_masked(tar_img, prd_img, cr1), compute_ssim_masked(tar_img, prd_img, cr1)


summary = {}
for dataset in ['GoPro', 'HIDE']:
    gt_dir = DATA_ROOT / 'test' / dataset / 'target'
    pred_dir = RESULT_ROOT / dataset
    gt_files = natsorted(glob(str(gt_dir / '*.png')) + glob(str(gt_dir / '*.jpg')))
    pred_files = natsorted(glob(str(pred_dir / '*.png')) + glob(str(pred_dir / '*.jpg')))
    assert len(gt_files) == len(pred_files) and gt_files, f'Missing files for {dataset}'
    scores = [eval_pair_gopro_hide(pair) for pair in zip(gt_files, pred_files)]
    summary[dataset] = (
        sum(s[0] for s in scores) / len(scores),
        sum(s[1] for s in scores) / len(scores),
    )

for dataset in ['RealBlur_J', 'RealBlur_R']:
    gt_dir = DATA_ROOT / 'test' / dataset / 'target'
    pred_dir = RESULT_ROOT / dataset
    gt_files = natsorted(glob(str(gt_dir / '*.png')) + glob(str(gt_dir / '*.jpg')))
    pred_files = natsorted(glob(str(pred_dir / '*.png')) + glob(str(pred_dir / '*.jpg')))
    assert len(gt_files) == len(pred_files) and gt_files, f'Missing files for {dataset}'
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        scores = list(executor.map(eval_pair_realblur, zip(gt_files, pred_files)))
    summary[dataset] = (
        sum(s[0] for s in scores) / len(scores),
        sum(s[1] for s in scores) / len(scores),
    )

print('Final Results:')
for dataset in ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']:
    psnr, ssim = summary[dataset]
    print(f'  {dataset}: PSNR {psnr:.6f}, SSIM {ssim:.6f}')
PY
