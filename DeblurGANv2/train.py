import logging
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import pretrainedmodels.models.inceptionresnetv2
ir2 = sys.modules['pretrainedmodels.models.inceptionresnetv2']
if hasattr(ir2, 'pretrained_settings') and 'inceptionresnetv2' in ir2.pretrained_settings:
    for key in ir2.pretrained_settings['inceptionresnetv2']:
        if 'url' in ir2.pretrained_settings['inceptionresnetv2'][key]:
            ir2.pretrained_settings['inceptionresnetv2'][key]['url'] = 'https://huggingface.co/jscanvic/mirror/resolve/main/inceptionresnetv2-520b38e4.pth'

import contextlib
from functools import partial

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from schedulers import LinearDecay, WarmRestart
from fire import Fire

cv2.setNumThreads(0)

# Use Ampere+ tensor cores for fp32 matmul/conv (TF32) and let cuDNN pick the
# fastest conv algorithms for the fixed 256x256 input size. No-ops on older GPUs
# like P100, so safe to keep on everywhere.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']
        # How often (in iterations) to compute the slow CPU-side PSNR/SSIM metrics.
        self.metric_every = config.get('metric_every', 100)
        # Mixed precision: use bf16 autocast (no GradScaler needed, so it is safe
        # with the WGAN-GP gradient penalty's double backward). Skipped on GPUs
        # without bf16 support (e.g. P100), where it would not help anyway.
        self.use_amp = bool(config.get('amp', True)) and torch.cuda.is_available() \
            and torch.cuda.is_bf16_supported()

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def train(self):
        self._init_params()
        start_epoch = self._maybe_resume()
        print('AMP (bf16 autocast): {} | metrics every {} iters'.format(
            'ON' if self.use_amp else 'OFF', self.metric_every))
        for epoch in range(start_epoch, self.config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            is_best = self.metric_counter.update_best_model()
            self._save_checkpoint(epoch, is_best)
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            with self._autocast():
                outputs = self.netG(inputs)
            loss_D = self._update_d(outputs, targets)
            self.optimizer_G.zero_grad()
            with self._autocast():
                loss_content = self.criterionG(outputs, targets)
                loss_adv = self.adv_trainer.loss_g(outputs, targets)
                loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_D)
            # PSNR/SSIM are computed on CPU (skimage) and force a GPU->CPU sync,
            # so only sample them every `metric_every` iterations, not every step.
            if i % self.metric_every == 0:
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='train')
            tq.set_postfix(loss=self.metric_counter.loss_message())
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = self.config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            with torch.no_grad(), self._autocast():
                outputs = self.netG(inputs)
                loss_content = self.criterionG(outputs, targets)
                loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.item(), loss_content.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        with self._autocast():
            loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)

    def _save_checkpoint(self, epoch, is_best=False):
        exp = self.config['experiment_desc']
        state = {
            'epoch': epoch,
            'model': self.netG.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
            'scheduler_D': self.scheduler_D.state_dict(),
            'adv_trainer': self.adv_trainer.state_dict(),
            'best_metric': self.metric_counter.best_metric,
        }
        torch.save(state, 'last_{}.h5'.format(exp))
        if is_best:
            # keep best_*.h5 weights-only for compatibility with predict.py
            torch.save({'model': self.netG.state_dict()}, 'best_{}.h5'.format(exp))

    def _maybe_resume(self):
        resume_path = self.config.get('resume')
        if not resume_path and self.config.get('auto_resume', True):
            default_last = 'last_{}.h5'.format(self.config['experiment_desc'])
            if os.path.exists(default_last):
                resume_path = default_last
        if resume_path and os.path.exists(resume_path):
            start_epoch = self._load_checkpoint(resume_path)
            print('Resumed from {} -> starting at epoch {}'.format(resume_path, start_epoch))
            return start_epoch
        return 0

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cuda')
        self.netG.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint.get('epoch', -1) + 1

        # If we resume past warmup, the backbone must be unfrozen and the G optimizer/
        # scheduler rebuilt on the full parameter set *before* loading their states,
        # because the in-loop unfreeze (epoch == warmup_epochs) will not fire anymore.
        if self.warmup_epochs and start_epoch > self.warmup_epochs:
            self.netG.module.unfreeze()
            self.optimizer_G = self._get_optim(self.netG.parameters())
            self.scheduler_G = self._get_scheduler(self.optimizer_G)

        if 'optimizer_G' in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
            self.adv_trainer.load_state_dict(checkpoint['adv_trainer'])
            self.metric_counter.best_metric = checkpoint.get('best_metric', 0)
        else:
            print('Checkpoint has weights only (no optimizer state); '
                  'continuing with fresh optimizer/scheduler.')
        return start_epoch


def main(config_path='config/config.yaml', resume=None, auto_resume=True):
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # CLI overrides for resume behaviour (Fire maps these to --resume / --auto_resume)
    if resume is not None:
        config['resume'] = resume
    config['auto_resume'] = auto_resume

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader,
                             batch_size=batch_size,
                             num_workers=0 if os.environ.get('DEBUG') else cpu_count(),
                             shuffle=True, drop_last=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train()


if __name__ == '__main__':
    Fire(main)
