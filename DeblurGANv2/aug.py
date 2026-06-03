from typing import List

import albumentations as albu


def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(p=1.0),
                                     albu.ShiftScaleRotate(p=1.0),
                                     albu.Transpose(p=1.0),
                                     albu.OpticalDistortion(p=1.0),
                                     albu.ElasticTransform(p=1.0),
                                     ])
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, p=1.0),
               'center': albu.CenterCrop(size, size, p=1.0)}[crop]
    pad = albu.PadIfNeeded(size, size)

    pipeline = albu.Compose([aug_fn, pad, crop_fn], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.CoarseDropout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.Sharpen,
        'jpeg': albu.ImageCompression,
        'gray': albu.ToGray,
        'pixelize': albu.Downscale,
        # ToDo: partial gray
    }
    return d[name]


def get_corrupt_function(config: List[dict]):
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob') if 'prob' in aug_params else .5
        if cls == albu.CoarseDropout:
            if 'num_holes' in aug_params:
                aug_params['max_holes'] = aug_params.pop('num_holes')
            if 'max_h_size' in aug_params:
                aug_params['max_height'] = aug_params.pop('max_h_size')
            if 'max_w_size' in aug_params:
                aug_params['max_width'] = aug_params.pop('max_w_size')
        augs.append(cls(p=prob, **aug_params))

    augs = albu.OneOf(augs)

    def process(x):
        return augs(image=x)['image']

    return process

