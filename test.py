import numpy as np
import os
import h5py
import torch
import pickle
import nibabel as nib
from datetime import datetime

from torch.utils.data import DataLoader

from model import SR_q_DL
from utils import init_device_seed, load_args_test
from dataset import TestDataset
from preprocess import get_vals_vecs

if __name__ == '__main__':
    args = load_args_test()
    device = init_device_seed(1234, args.cuda_visible)
    vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr = get_vals_vecs()
    output_dir = "./result/" + datetime.now().strftime("%Y-%m-%d %H_%M_%S")

    test_list = os.listdir("./data/test_h5")
    model = SR_q_DL(36, 270).to(device)
    checkpoint = torch.load('./model/model_dict', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for idx, subject in enumerate(test_list):
        hf = h5py.File(f"./data/test_h5/{subject}/data.h5", "r")
        lr_vol = hf["lr"]
        mask_index = hf["mask_index"]
        hr_e_vol = np.zeros((270, 144+8, 174+8, 144+8), dtype=np.float32)

        dataset = TestDataset(lr_vol, mask_index)
        test_loader = DataLoader(dataset, batch_size=args.batch_size)

        for idx, (lr, mask) in enumerate(test_loader):
            lr = lr.to(device, dtype=torch.float32)
            hr_e = model(lr).detach().cpu().numpy()
            
            for batch_idx in range(mask.shape[0]):
                x, y, z = mask[batch_idx]
                hr_e_vol[:, x*2:x*2+2, y*2:y*2+2, z*2:z*2+2] = hr_e[batch_idx]
            print(f'\r{idx}/{len(test_loader)}', end='')

        hr_b0s = hf["hr_b0"]
        dwi_b0 = np.mean(hr_b0s, axis=3)
        np.clip(hr_e_vol, 0, 1, out=hr_e_vol)
        for dif in range(270):
            hr_e_vol[dif] *= dwi_b0

        hr_vol = np.empty((288, 144+8+1, 174+8, 144+8+1), dtype=np.float32)
        hr_vol[dif_indexes_hr, :-1, :, :-1] = hr_e_vol[:270]
        hr_vol[dif_indexes_0, :-1, :, :-1] = np.transpose(hr_b0s, (3, 0, 1, 2))
        hr_vol = np.transpose(hr_vol, (1, 2, 3, 0))

        with open(f"./data/test_h5/{subject}/header", "rb") as f:
            dwi_header = pickle.load(f)

        os.makedirs(f'{output_dir}/{subject}', exist_ok=True)
        ni_img = nib.Nifti1Image(hr_vol, None, header=dwi_header)
        nib.save(ni_img, f'{output_dir}/{subject}/data.nii.gz')