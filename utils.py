import torch
import numpy as np
import random
import os
import argparse


def init_device_seed(seed, cuda_visible):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return device

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=bool, default=False, help='loading pretrained model')
    parser.add_argument('--cuda_visible', default='0', help='set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', type=int, default=128, help='set batch size')
    args = parser.parse_args()
    return args

def load_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_visible', default='0', help='set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_path', default='./data/val', help='set validation path')
    parser.add_argument('--batch_size', type=int, default=128, help='set batch size')
    args = parser.parse_args()
    return args