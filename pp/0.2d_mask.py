import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import timm
import nibabel
import scipy.ndimage as ndi
from skimage.io import imread
from multiprocessing.pool import Pool

DEBUG = False

path = os.path.dirname(os.path.realpath(__file__))
masks = glob.glob(path + '/../ts/organ/*')

if DEBUG:
    masks = masks[:100]

print(path, len(masks))


def process(mask_path):
    patient, study = mask_path.split('/')[-1].split('_')
    spleen = nibabel.load(path + f'/../ts/organ/{patient}_{study}/spleen.nii.gz').get_fdata()
    spleen = np.flip(np.rollaxis(spleen, 0, 2), 0)
    kidney_l = nibabel.load(path + f'/../ts/organ/{patient}_{study}/kidney_left.nii.gz').get_fdata()
    kidney_l = np.flip(np.rollaxis(kidney_l, 0, 2), 0)
    kidney_r = nibabel.load(path + f'/../ts/organ/{patient}_{study}/kidney_right.nii.gz').get_fdata()
    kidney_r = np.flip(np.rollaxis(kidney_r, 0, 2), 0)
    small_bowel = nibabel.load(path + f'/../ts/organ/{patient}_{study}/small_bowel.nii.gz').get_fdata()
    small_bowel = np.flip(np.rollaxis(small_bowel, 0, 2), 0)
    liver = nibabel.load(path + f'/../ts/organ/{patient}_{study}/liver.nii.gz').get_fdata()
    liver = np.flip(np.rollaxis(liver, 0, 2), 0)
    merged = spleen + kidney_l + kidney_r + small_bowel + liver
    r = merged.reshape(-1, merged.shape[-1]).sum(0)
    
    gap = int(np.floor(liver.shape[2] / 50))
    msk = np.zeros((spleen.shape[0], spleen.shape[1], 4), dtype=bool)
    for idx in range(0, liver.shape[2], gap):
        if r[idx] == 0:
            continue
        msk[:, :, 0] = spleen[:, :, idx]
        msk[:, :, 1] = kidney_l[:, :, idx] + kidney_r[:, :, idx]
        msk[:, :, 2] = small_bowel[:, :, idx]
        msk[:, :, 3] = liver[:, :, idx]
        np.save(path + f'/../input/2d_mask/{patient}_{study}_{idx:04}', msk)


def process_body(mask_path):
    patient, study = mask_path.split('/')[-1].split('_')
    merged = nibabel.load(path + f'/../ts/body/{patient}/{study}/body.nii.gz').get_fdata()
    merged = np.flip(np.rollaxis(merged, 0, 2), 0)
    r = merged.reshape(-1, merged.shape[-1]).sum(0)
    
    gap = int(np.floor(merged.shape[2] / 10))
    msk = np.zeros((merged.shape[0], merged.shape[1], 1), dtype=bool)
    for idx in range(0, merged.shape[2], gap):
        if r[idx] == 0:
            continue
        msk[:, :, 0] = merged[:, :, idx]
        np.save(path + f'/../input/2d_mask_body/{patient}_{study}_{idx:04}', msk)
        
        
def process_organ():
    p = Pool(processes=8)
    for mask_path in tqdm.tqdm(masks):
        p.apply_async(process, (mask_path, ))
    p.close()
    p.join()
    


def mask_qc(mask_path):
    patient, study = mask_path.split('/')[-1].split('_')
    merged = nibabel.load(path + f'/../ts/body/{patient}/{study}/body.nii.gz').get_fdata()
    label, n = ndi.label(merged[:, :, merged.shape[-1]//2])
    return n

print('############## STEP 1 ################')
process_organ()

print('############## STEP 2 ################')
r = []
p = Pool(processes=8)
for mask_path in tqdm.tqdm(masks):
    r.append(p.apply_async(mask_qc, (mask_path, )))
p.close()
p.join()
r = [e.get() for e in r]

# filter out wrong mask
qc_df = pd.DataFrame({'file': masks, 'area': r})
used = qc_df.sort_values('area').head(4000)

# emploid = set(used.file.str.split('/', expand=True)[7].values)
emploid = set([x.split('/')[-1] for x in used.file.values])

# print(qc_df.shape, len(emploid))

print('############## STEP 3 ################')
others = glob.glob(path + f'/../input/2d_mask/*.npy')
to_remove = []
for o in others:
    if not '_'.join(o.split('/')[-1].split('_')[:2]) in emploid:
        to_remove.append(o)
        
print('Ratio of removal series: {}/{}'.format(len(to_remove), len(others)))

for tr in to_remove:
    os.remove(tr)
    
print('############# STEP 4 #################')

# process_body(used.file.values[0])
p = Pool(processes=12)
for mask_path in tqdm.tqdm(used.file.values):
    p.apply_async(process_body, (mask_path, ))
p.close()
p.join()