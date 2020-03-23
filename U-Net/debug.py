import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data2/imgs/'
dir_mask = 'data2/masks/'
dir_checkpoint = 'checkpoints/'


epochs=5
batch_size=15
lr=0.01
val_percent=0.1
save_cp=True
img_scale=1

dataset = BasicDataset(dir_img, dir_mask, img_scale)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for batch in train_loader:

    imgs = batch['image'].type(dtype)
    true_masks = batch['mask']
    print("imgs shape: ", imgs.shape)
    print("mask shape: ", true_masks.shape)
    print("imgs type: ", type(imgs))
    print("mask type: ", type(true_masks))