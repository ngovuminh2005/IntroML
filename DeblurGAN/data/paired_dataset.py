import os.path
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class PairedDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'input')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'target')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        assert len(self.A_paths) == len(self.B_paths), \
            'input (%d) and target (%d) counts differ' % (len(self.A_paths), len(self.B_paths))
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if 'resize' in self.opt.resize_or_crop:
            A_img = A_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
            B_img = B_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        A = self.transform(A_img)
        B = self.transform(B_img)
        h, w, fs = A.size(1), A.size(2), self.opt.fineSize
        h_offset = random.randint(0, max(0, h - fs - 1))
        w_offset = random.randint(0, max(0, w - fs - 1))
        A = A[:, h_offset:h_offset + fs, w_offset:w_offset + fs]
        B = B[:, h_offset:h_offset + fs, w_offset:w_offset + fs]
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = torch.LongTensor([i for i in range(A.size(2) - 1, -1, -1)])
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'PairedDataset'
