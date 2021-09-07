import numpy as np
import os
import nibabel as nib
import h5py
import pickle
import itertools

def get_vals_vecs():
    subject = os.listdir("./data/train")[0]

    with open(f"./data/train/{subject}/bvals", "r") as f:
        vals = np.array(list(map(int, f.readline().split())))

    with open(f"./data/train/{subject}/bvecs", "r") as f:
        vecs = np.empty((288, 3))
        for i in range(3):
            vecs[:, i] = list(map(float, f.readline().split()))

    dif_indexes_0 = np.where(vals < 100)[0]
    dif_indexes_hr = np.where(vals > 100)[0]
    with open(f"./data/diffusion_indexes_train", "rb") as f:
        dif_indexes_lr = pickle.load(f)
    
    return vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr

def make_lr(dwi, mask, dif_indexes_lr):
    lr = np.zeros((72+4, 87+4, 72+4, 36), dtype=np.float32)
    mask_lr = np.zeros((72+4, 87+4, 72+4), dtype=np.uint8)
    for i, j, k in itertools.product(range(72+2), range(87+2), range(72+2)):
        lr[i, j, k] = np.mean(dwi[i*2:i*2+2, j*2:j*2+2, k*2:k*2+2, dif_indexes_lr], axis=(0,1,2))
        mask_lr[i, j, k] = np.max(mask[i*2:i*2+2, j*2:j*2+2, k*2:k*2+2])
    return lr, mask_lr

def make_train_dataset():
    train_list = os.listdir("./data/train")
    train_hrs = np.zeros((5, 144+8, 174+8, 144+8, 270), dtype=np.float32)
    train_lrs = np.zeros((5, 72+4, 87+4, 72+4, 36), dtype=np.float32)
    mask_lrs = np.zeros((5, 72+4, 87+4, 72+4), dtype=np.uint8)

    vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr = get_vals_vecs()

    for idx, subject in enumerate(train_list):
        dwi = nib.load(f"./data/train/{subject}/data.nii.gz")
        dwi = np.array(dwi.get_fdata(), dtype=np.float32)[:-1, :, :-1]
        mask = nib.load(f"./data/train/{subject}/nodif_brain_mask.nii.gz")
        mask = np.array(mask.get_fdata(), dtype=np.uint8)[:-1, :, :-1]

        dwi = np.pad(dwi, ((4, 4), (4, 4), (4, 4), (0, 0)), "constant", constant_values=0)
        mask = np.pad(mask, ((4, 4), (4, 4), (4, 4)), "constant", constant_values=0)

        dwi_b0 = np.mean(dwi[..., dif_indexes_0], axis=3)

        for dif in range(288):
            dwi[..., dif] /= dwi_b0
        np.nan_to_num(dwi, copy=False)
        np.clip(dwi, 0, 1, out=dwi)

        train_hrs[idx] = dwi[..., dif_indexes_hr]
        train_lrs[idx], mask_lrs[idx] = make_lr(dwi, mask, dif_indexes_lr)

    mask_index = np.array(np.where(mask_lrs == 1)).T

    hf = h5py.File("./data/train.h5", "w")
    hf.create_dataset("hr", data=train_hrs)
    hf.create_dataset("lr", data=train_lrs)
    hf.create_dataset("mask", data=mask_index)
    hf.close()

def make_test_dataset():
    test_list = os.listdir("./data/test")
    vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr = get_vals_vecs()

    for idx, subject in enumerate(test_list):
        dwi = nib.load(f"./data/test/{subject}/data.nii.gz")
        dwi_header = dwi.header.copy()
        dwi = np.array(dwi.get_fdata(), dtype=np.float32)[:-1, :, :-1]
        mask = nib.load(f"./data/test/{subject}/nodif_brain_mask.nii.gz")
        mask = np.array(mask.get_fdata(), dtype=np.uint8)[:-1, :, :-1]

        dwi = np.pad(dwi, ((4, 4), (4, 4), (4, 4), (0, 0)), "constant", constant_values=0)
        mask = np.pad(mask, ((4, 4), (4, 4), (4, 4)), "constant", constant_values=0)

        dwi_b0 = np.mean(dwi[..., dif_indexes_0], axis=3)
        dwi_b0s = dwi[..., dif_indexes_0]

        for dif in range(288):
            dwi[..., dif] /= dwi_b0
        np.nan_to_num(dwi, copy=False)
        np.clip(dwi, 0, 1, out=dwi)

        test_lr, mask_lr = make_lr(dwi, mask, dif_indexes_lr)
        mask_index = np.array(np.where(mask_lr == 1)).T
        print(np.where(mask_lr == 1))

        os.makedirs(f"./data/test_h5/{subject}", exist_ok=True)
        hf = h5py.File(f"./data/test_h5/{subject}/data.h5", "w")
        hf.create_dataset("lr", data=test_lr)
        hf.create_dataset("mask_index", data=mask_index)
        hf.create_dataset("mask_hr", data=mask)
        hf.close()

        with open(f"./data/test_h5/{subject}/header", "wb") as f:
            pickle.dump(dwi_header, f)
        break

if __name__ == "__main__":
    make_train_dataset()
    make_test_dataset()