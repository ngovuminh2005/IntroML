"""
Evaluate a trained DeblurGANv2 generator on a test folder and report mean
PSNR / SSIM.

Expected folder layout (filenames must match between the two subfolders):

    test_dir/
        input/         <- blurred images (model input)
        groundtruth/   <- sharp reference images

Usage (PowerShell):

    python eval_folder.py --test_dir path\to\test --weights_path fpn_inception.h5
    # optionally save deblurred results:
    python eval_folder.py --test_dir path\to\test --weights_path fpn_inception.h5 --out_dir results\

Metrics are computed exactly as in training (models/models.py):
PSNR from util.metrics, SSIM from skimage with data_range=255.
"""
import inspect
import os
from glob import glob

import cv2
import numpy as np
from fire import Fire
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm

from predict import Predictor
from util.metrics import PSNR

# Same skimage-version handling as models/models.py.
_SSIM_USES_CHANNEL_AXIS = 'channel_axis' in inspect.signature(SSIM).parameters


def _compute_ssim(fake, real):
    if _SSIM_USES_CHANNEL_AXIS:
        return SSIM(fake, real, channel_axis=-1, data_range=255)   # skimage >= 0.19
    return SSIM(fake, real, multichannel=True, data_range=255)     # skimage < 0.19


def _list_images(folder):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files = []
    for ext in exts:
        files += glob(os.path.join(folder, ext))
        files += glob(os.path.join(folder, ext.upper()))
    return {os.path.basename(f): f for f in files}


def main(test_dir: str,
         weights_path: str = 'fpn_inception.h5',
         input_subdir: str = 'input',
         gt_subdir: str = 'target',
         out_dir: str = None):
    input_dir = os.path.join(test_dir, input_subdir)
    gt_dir = os.path.join(test_dir, gt_subdir)

    inputs = _list_images(input_dir)
    gts = _list_images(gt_dir)

    common = sorted(set(inputs) & set(gts))
    if not common:
        raise SystemExit(
            f'No matching filenames between "{input_dir}" and "{gt_dir}".')
    missing = sorted((set(inputs) | set(gts)) - set(common))
    if missing:
        print(f'Warning: skipping {len(missing)} unpaired file(s), '
              f'e.g. {missing[:5]}')

    predictor = Predictor(weights_path=weights_path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    psnr_sum, ssim_sum = 0.0, 0.0
    for name in tqdm(common, desc='Evaluating'):
        blur = cv2.cvtColor(cv2.imread(inputs[name]), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(gts[name]), cv2.COLOR_BGR2RGB)

        pred = predictor(blur, None)  # RGB uint8, same H,W as input

        # Guard against any off-by-a-pixel size mismatch with the groundtruth.
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        psnr_sum += PSNR(pred, gt)
        ssim_sum += _compute_ssim(pred, gt)

        if out_dir:
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    n = len(common)
    print(f'\nEvaluated {n} image(s)')
    print(f'PSNR = {psnr_sum / n:.4f}')
    print(f'SSIM = {ssim_sum / n:.4f}')


if __name__ == '__main__':
    Fire(main)
