import os

import base64
import typing as t
import zlib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imsave, imread
import numpy as np
import pandas as pd
import os

from typing import Callable, List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import skimage
import random
import tifffile
import torch
import math
import glob
import ast
import torchvision
import albumentations
from pathlib import Path

import segmentation_models_pytorch as smp
from torch import nn
import torch
import tqdm
import cv2

path = os.path.dirname(os.path.realpath(__file__))


files = glob.glob(path + '/../input/2d_mask_body/*.npy')
file_df = pd.DataFrame({'file': files})

print(file_df.shape[0])

n = len(file_df.file.iloc[0].split('/')) - 1

file_df['patient'] = file_df.file.str.split('/', expand=True)[n].str.slice(0, -4).str.split('_', expand=True)[0]
file_df['study'] = file_df.file.str.split('/', expand=True)[n].str.slice(0, -4).str.split('_', expand=True)[1]
file_df['idx'] = file_df.file.str.split('/', expand=True)[n].str.slice(0, -4).str.split('_', expand=True)[2]
file_df['fold'] = -1

file_df.loc[file_df.study.isin(file_df.study.value_counts().index[:3600]), 'fold'] = 1
file_df.loc[~file_df.study.isin(file_df.study.value_counts().index[:3600]), 'fold'] = 0

print(file_df.fold.value_counts())


image_size=512

transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
#     albumentations.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=(0,0,0)),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
])

transforms_valid = albumentations.Compose([
#     albumentations.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=(0,0,0))
    albumentations.Resize(image_size, image_size),
])


class SEGDataset(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.tfms = tfms
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path('.')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # print(str(self.path / '..' / 'input' / 'fish_crops' / '{}.png'.format(row.object_id)))
        img = imread(row.file.replace('npy', 'png').replace('2d_mask_body', 'pngs'))
        mask = np.load(row.file).astype(int)
        # print(mask.shape)
        if self.tfms:
            tf = self.tfms(image=img, mask=mask)
            img = tf['image']
            mask = tf['mask'] > 0
        # if not img.shape[0] == 512 or not img.shape[1] == 512:
        #     img = cv2.resize(img, (512, 512))
        #     mask = cv2.resize((mask * 255.0).astype(np.uint8), (512, 512)) > 0
        img = np.stack([img] * 3, -1)
        img = self.tensor_tfms(img)
        
        mask = np.rollaxis(mask, 2, 0)
        return img, torch.tensor(mask)
    
    
cd = SEGDataset(file_df[file_df.fold == 1].sample(frac=1), tfms=transforms_train)
train_dl = DataLoader(cd, num_workers=16, batch_size=32)

val_ds = SEGDataset(file_df[file_df.fold != 1].sample(frac=1), tfms=transforms_valid)
valid_dl = DataLoader(val_ds, num_workers=16, batch_size=32)

print('######################### Step1 #################################')
model = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).cuda()


# loss_function = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([4, 4, 4, 4]).cuda())
loss_function = smp.losses.FocalLoss('multilabel')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

model.train()
for epoch in range(1):
    _ = model.train()
    tq = tqdm.tqdm(train_dl)
    train_loss = []
    for iters, (images, targets) in enumerate(tq):
        images = images.cuda()
        targets = targets.cuda().float()
        with torch.cuda.amp.autocast():
            logits = model(images)
        loss = loss_function(logits.float(), targets)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss1.backward()
        # optimizer.step()
        tq.set_postfix(loss=np.mean(train_loss))
    
    _ = model.eval()
    ious = []
    for image, targets in tqdm.tqdm(valid_dl):
        with torch.no_grad():
            r = (model(image.cuda().flip(-1)).flip(-1) + model(image.cuda())) / 2
            r = torch.sigmoid(r).cpu()

            ious.append(((r > 0.5) * targets).sum() / (((r > 0.5) + targets) > 0.5).sum())
    print(f'Epochs: {epoch}, iou: {np.mean(ious)}')

torch.save(model.state_dict(), path + '/../input/body_model_v2.pth')


