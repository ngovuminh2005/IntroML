import inspect

import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR

# Older scikit-image releases accept **kwargs and use `multichannel`, so an
# unknown `channel_axis` is silently ignored (no TypeError) and SSIM runs in
# single-channel mode -> "win_size exceeds image extent". Pick the right kwarg
# by inspecting the signature instead of relying on a TypeError fallback.
_SSIM_USES_CHANNEL_AXIS = 'channel_axis' in inspect.signature(SSIM).parameters


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        if _SSIM_USES_CHANNEL_AXIS:
            ssim = SSIM(fake, real, channel_axis=-1, data_range=255)   # skimage >= 0.19
        else:
            ssim = SSIM(fake, real, multichannel=True, data_range=255)  # skimage < 0.19
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
