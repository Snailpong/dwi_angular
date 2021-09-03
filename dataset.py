from torch.utils.data import DataLoader
import os
import random
import torch
import h5py
import numpy as np

class TrainDataset(DataLoader):
    def __init__(self):
        hf = h5py.File('./data/train.h5', 'r')
        self.hr = hf['hr']
        self.lr = hf['lr']
        self.mask = hf['mask']

    def __getitem__(self, index):
        idx, x, y, z = self.mask[index]
        patch_hr = torch.from_numpy(self.hr[idx, x*2:x*2+2, y*2:y*2+2, z*2:z*2+2])
        patch_lr = torch.from_numpy(self.lr[idx, x-2:x+3, y-2:y+3, z-2:z+3])

        return patch_lr.permute(3, 0, 1, 2), patch_hr.permute(3, 0, 1, 2)

    def __len__(self):
        return self.mask.shape[0]


class TestDataset(DataLoader):
    def __init__(self, lr, mask):
        self.lr = lr
        self.mask = mask

    def __getitem__(self, index):
        x, y, z = self.mask
        patch_lr = torch.from_numpy(self.lr[x-2:x+3, y-2:y+3, z-2:z+3])

        return patch_lr.permute(3, 0, 1, 2)

    def __len__(self):
        return self.mask.shape[0]