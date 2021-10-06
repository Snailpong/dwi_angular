import torch
import os
import time
import numpy as np

from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.data.dataset import random_split

from model import SR_q_DL
from utils import init_device_seed, load_args
from dataset import TrainDataset

def train():
    args = load_args()
    device = init_device_seed(1234, args.cuda_visible)

    dataset = TrainDataset()
    len_dataset = dataset.__len__()
    len_train_val = [int(len_dataset * 0.9), len_dataset - int(len_dataset * 0.9)]

    train_dataset, val_dataset = random_split(dataset, len_train_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    os.makedirs('./model', exist_ok=True)
    model = SR_q_DL(36, 288).to(device)
    epoch = 0

    if args.load_model:
        checkpoint = torch.load('./model/model_dict', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    criterion_mse = nn.MSELoss()

    while epoch < 20:
        timer = time.time()
        epoch += 1

        train_loss = .0
        val_loss = .0

        # train
        model.train()

        for idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, dtype=torch.float32)
            hr = hr.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            hr_e = model(lr)
            loss = criterion_mse(hr_e, hr)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Loss display
            train_loss += loss.item()
            print(f'\rEpoch {epoch} {idx}/{len(train_loader)} Train loss: {np.around(train_loss / (idx + 1), 4)}\t', end='')

        # validation
        model.eval()
        print('Val loss: ', end='')

        for idx, (lr, hr) in enumerate(val_loader):
            lr = lr.to(device, dtype=torch.float32)
            hr = hr.to(device, dtype=torch.float32)

            hr_e = model(lr)
            loss = criterion_mse(hr_e, hr)
            val_loss += loss.item()

        print(f'{np.around(val_loss / (idx + 1), 4)}, Time: {time.time() - timer}s')

        # Save checkpoint per epoch
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
        }, './model/model_dict')

if __name__ == '__main__':
    train()