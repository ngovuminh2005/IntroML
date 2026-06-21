"""
Deblur images with a trained DeblurGAN generator and (optionally) compute PSNR / SSIM.

Examples
--------
# Single image, no ground truth (just deblur + save):
python deblur_eval.py --name experiment_name --blurred path/to/blur.png --gpu_ids -1

# Single image WITH ground truth (deblur + PSNR/SSIM):
python deblur_eval.py --name experiment_name \
    --blurred path/to/blur.png --sharp path/to/sharp.png --gpu_ids -1

# Whole dataset (folders with matching filenames):
python deblur_eval.py --name experiment_name \
    --blurred_dir path/to/blurA --sharp_dir path/to/sharpB --gpu_ids -1
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from models import networks

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def build_args():
    p = argparse.ArgumentParser()
    # generator / checkpoint
    p.add_argument('--name', type=str, default='experiment_name',
                   help='experiment name -> ./checkpoints/<name>/<which_epoch>_net_G.pth')
    p.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    p.add_argument('--which_epoch', type=str, default='latest')
    p.add_argument('--which_model_netG', type=str, default='resnet_9blocks')
    p.add_argument('--input_nc', type=int, default=3)
    p.add_argument('--output_nc', type=int, default=3)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--norm', type=str, default='instance')
    p.add_argument('--learn_residual', action='store_true', default=True,
                   help='DeblurGAN is trained with residual learning; keep True for the released weights')
    p.add_argument('--gpu_ids', type=str, default='0', help='e.g. 0 ; use -1 for CPU')
    # data
    p.add_argument('--blurred', type=str, default=None, help='single blurry image')
    p.add_argument('--sharp', type=str, default=None, help='single ground-truth sharp image (optional)')
    p.add_argument('--blurred_dir', type=str, default=None, help='folder of blurry images')
    p.add_argument('--sharp_dir', type=str, default=None, help='folder of ground-truth sharp images (optional)')
    p.add_argument('--results_dir', type=str, default='./results/deblur_eval')
    p.add_argument('--pad', type=int, default=4,
                   help='generator needs H,W divisible by this; we pad then crop back')
    return p.parse_args()


# ---------- image <-> tensor ----------
def load_image_tensor(path, pad_to):
    """PIL RGB -> tensor in [-1, 1], shape (1,3,H,W). Returns tensor and original (H,W)."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32)            # H,W,3 in [0,255]
    h, w = arr.shape[:2]
    arr = arr / 255.0 * 2.0 - 1.0                        # -> [-1,1]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    # pad bottom/right so H,W divisible by pad_to (reflect avoids border artifacts)
    ph = (pad_to - h % pad_to) % pad_to
    pw = (pad_to - w % pad_to) % pad_to
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (h, w)


def tensor_to_uint8(t, orig_hw):
    """tensor (1,3,H,W) in [-1,1] -> uint8 H,W,3, cropped back to original size."""
    h, w = orig_hw
    arr = t[0].detach().cpu().float().numpy()
    arr = (arr.transpose(1, 2, 0) + 1) / 2.0 * 255.0
    arr = arr[:h, :w, :]
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------- metrics ----------
def psnr(a, b):
    """a, b uint8 arrays."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(a, b):
    """a, b uint8 arrays. Uses skimage if available, else a simple global SSIM."""
    if _HAS_SKIMAGE:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    # fallback: global (non-windowed) SSIM on grayscale
    x = a.astype(np.float64).mean(2)
    y = b.astype(np.float64).mean(2)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mx * my + c1) * (2 * cov + c2)) /
                 ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def load_generator(opt, gpu_ids):
    netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                             opt.which_model_netG, opt.norm, False,
                             gpu_ids, False, opt.learn_residual)
    ckpt = os.path.join(opt.checkpoints_dir, opt.name, '%s_net_G.pth' % opt.which_epoch)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError('Generator weights not found: %s' % ckpt)
    state = torch.load(ckpt, map_location='cpu')
    netG.load_state_dict(state)
    netG.eval()
    print('Loaded generator from %s' % ckpt)
    return netG


def deblur_one(netG, device, blur_path, sharp_path, out_dir, pad):
    inp, orig = load_image_tensor(blur_path, pad)
    inp = inp.to(device)
    with torch.no_grad():
        out = netG(inp)
    fake = tensor_to_uint8(out, orig)

    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(blur_path))[0] + '_deblurred.png'
    out_path = os.path.join(out_dir, out_name)
    Image.fromarray(fake).save(out_path)
    print('Saved %s' % out_path)

    if sharp_path and os.path.isfile(sharp_path):
        gt = np.asarray(Image.open(sharp_path).convert('RGB'))[:orig[0], :orig[1], :]
        return psnr(fake, gt), ssim(fake, gt)
    return None, None


def main():
    opt = build_args()
    gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if int(x) >= 0]
    device = torch.device('cuda:%d' % gpu_ids[0] if gpu_ids else 'cpu')
    netG = load_generator(opt, gpu_ids)
    if not gpu_ids:
        netG = netG.cpu()

    if not _HAS_SKIMAGE:
        print('[warn] scikit-image not installed -> using a crude fallback SSIM. '
              'Run `pip install scikit-image` for proper SSIM.')

    # build (blur, sharp) pair list
    pairs = []
    if opt.blurred:
        pairs.append((opt.blurred, opt.sharp))
    elif opt.blurred_dir:
        names = sorted(f for f in os.listdir(opt.blurred_dir) if f.lower().endswith(IMG_EXTS))
        for n in names:
            bp = os.path.join(opt.blurred_dir, n)
            sp = os.path.join(opt.sharp_dir, n) if opt.sharp_dir else None
            pairs.append((bp, sp))
    else:
        raise ValueError('Provide --blurred or --blurred_dir')

    psnrs, ssims = [], []
    for bp, sp in pairs:
        ps, ss = deblur_one(netG, device, bp, sp, opt.results_dir, opt.pad)
        if ps is not None:
            psnrs.append(ps); ssims.append(ss)
        else:
            print('%-40s  (deblurred, no ground truth)' % os.path.basename(bp))

    if psnrs:
        print('\n=== Average over %d images ===' % len(psnrs))
        print('PSNR = %.4f dB' % (sum(psnrs) / len(psnrs)))
        print('SSIM = %.4f' % (sum(ssims) / len(ssims)))


if __name__ == '__main__':
    main()
