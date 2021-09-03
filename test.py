import numpy as np
import os
import nibabel as nib
import itertools

from model import SR_q_DL
from utils import init_device_seed, load_args
from dataset import TrainDataset
from preprocess import get_vals_vecs

if __name__ == '__main__':
    args = load_args()
    device = init_device_seed(1234, args.cuda_visible)
    vals, vecs, dif_indexes_0, dif_indexes_hr, dif_indexes_lr = get_vals_vecs()

    test_list = os.listdir("./data/test")
    model = SR_q_DL(36, 270).to(device)
    model.eval()

    for idx, subject in enumerate(test_list):
        
        print(mask_index.shape)

        dataset = TestDataset(lr, mask_index)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        for idx, lr in enumerate(test_loader):
            lr = lr.to(device, dtype=torch.float32)
            hr_e = model(lr).detach().cpu()

            break
        break

