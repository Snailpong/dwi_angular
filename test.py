import numpy as np
import os
import h5py
import torch
import nibabel as nib
import itertools
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from model import SR_q_DL
from utils import init_device_seed, load_args_test
from dataset import TestDataset
from preprocess import get_vals_vecs

if __name__ == '__main__':
    args = load_args_test()
    device = init_device_seed(1234, args.cuda_visible)
    vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr = get_vals_vecs()

    test_list = os.listdir("./data/test_h5")
    model = SR_q_DL(36, 270).to(device)
    checkpoint = torch.load('./model/model_dict', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for idx, subject in enumerate(test_list):
        hf = h5py.File(f"./data/test_h5/{subject}/data.h5", "r")
        lr_vol = hf["lr"]
        mask_index = hf["mask_index"]
        # hr_e_vol = np.zeros((72+4, 87+4, 72+4, 270, 2, 2, 2), dtype=np.float32)
        hr_e_vol = np.zeros((144+8, 174+8, 144+8, 270), dtype=np.float32)
        # hr_e_vol = np.empty((mask_index.shape[0], 270, 2, 2, 2))

        print(mask_index.shape)

        dataset = TestDataset(lr_vol, mask_index)
        test_loader = DataLoader(dataset, batch_size=args.batch_size)

        for idx, (lr, mask) in enumerate(test_loader):
            lr = lr.to(device, dtype=torch.float32)
            # print(lr.min(), lr.max())
            hr_e = model(lr).detach().cpu().numpy()
            # print(hr_e[0,0].ravel())
            
            for batch_idx in range(mask.shape[0]):
                x, y, z = mask[batch_idx]
                hr_e_vol[x*2:x*2+2, y*2:y*2+2, z*2:z*2+2] = np.transpose(hr_e[batch_idx], (1, 2, 3, 0))
                # hr_e_vol[x, y, z] = np.transpose(hr_e[batch_idx], (1, 2, 3, 0))
            print(f'\r{idx}/{len(test_loader)}', end='')
            # hr_e_vol[idx*args.batch_size:min((idx+1)*args.batch_size, mask_index.shape[0])] = hr_e

        # hr_e_vol = np.transpose(hr_e_vol, (0, 4, 1, 5, 2, 6, 3))
        # hr_e_vol = np.transpose(hr_e_vol, (1, 2, 3, 0))
        # print(hr_e_vol.shape)
        # hr_e_vol = np.reshape(hr_e_vol, (144+8, 174+8, 144+8, 270))
        print(hr_e_vol.shape)
        print(hr_e_vol.max(), hr_e_vol.min())

        plt.subplot(1, 3, 1)
        plt.imshow(hr_e_vol[72, :, :, 0])
        plt.subplot(1, 3, 2)
        plt.imshow(hr_e_vol[72, :, :, 1])
        plt.subplot(1, 3, 3)
        plt.show()

