import matplotlib.pyplot as plt
import os
import cv2
import glob
import gdcm
import pydicom
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy import ndimage as ndi
import nibabel
from tqdm import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut
import segmentation_models_pytorch as smp
from torch import nn
import albumentations
import json
from multiprocessing.pool import Pool
import torchvision
import torch
import SimpleITK as sitk
import pickle as pk



def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2    

    pixel_array = (pixel_array * slope) + intercept

    # HU filter to isolate the body region
    # min_hu = -150
    # max_hu = 1000
    # body_mask = np.where((pixel_array > min_hu) & (pixel_array < max_hu), 1, 0)
    # pixel_array = pixel_array * body_mask

    # Window-level clipping
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array


def load_dicom_series(directory_path):
    # Get the list of all the DICOM filenames in the directory
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
    reader.SetFileNames(dicom_names)
    
    # Load them into a single volume and return
    image = reader.Execute()
    return image

def pydicom_load_series(dir_path):
    imgs = {}
    for f in sorted(glob.glob(dir_path + '*')):
        dicom = pydicom.dcmread(f)
        pos_z = dicom[(0x20, 0x32)].value[-1]
        img = standardize_pixel_array(dicom)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        imgs[pos_z] = img
    img_lst = sorted([(k, v) for k, v in imgs.items()], key=lambda x: x[0], reverse=False)
    arr = np.stack([x[1] for x in img_lst], -1)
    original_spacing_A = (float(dicom.PixelSpacing[0]), float(dicom.PixelSpacing[1]), float(dicom.SliceThickness))
    return arr, original_spacing_A, [float(x) for x in imgs.keys()]

import scipy.ndimage

def resample_volume(volume, original_spacing, target_spacing):
    # Calculate the resampling factor
    zoom_factors = [
        orig_spac/targ_spac for orig_spac, targ_spac in zip(original_spacing, target_spacing)
    ]

    # Resample the volume using scipy's ndimage.zoom function
    resampled_volume = scipy.ndimage.zoom(volume, zoom_factors, order=1)  # order=1 represents bilinear interpolation

    return resampled_volume


def resample_image_to_spacing(image, new_spacing=[1, 1, 1], interpolator=sitk.sitkLinear):
    # Calculate the new image size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for ospc, nspc, osz in zip(original_spacing, new_spacing, original_size)]

    # Resampling (note: 'Transform' is set to identity by default)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(interpolator)
    
    resampled_image = resample.Execute(image)
    return resampled_image


tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

path = os.path.dirname(os.path.realpath(__file__))

# loading model

model = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).cuda()

model.load_state_dict(torch.load(path + f'/../input/body_model_v2.pth'))
_ = model.eval()


four_model = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
).cuda()

four_model.load_state_dict(torch.load(path + f'/../input/4_classes_seg_model_v2.pth'))
_ = four_model.eval()



if not os.path.exists(path + '/../input/cache_100'):
    os.mkdir(path + '/../input/cache_100')
    
rescaled_path = path + '/../input/cache_100/rescaled'
mask_path = path + '/../input/cache_100/masks'

if not os.path.exists(rescaled_path):
    os.mkdir(rescaled_path)
    
if not os.path.exists(mask_path):
    os.mkdir(mask_path)

def process(pth):
    arr2, os, zaxis = pydicom_load_series(pth)
    target_spacing = (1, 1, 5)  # For example, 1mm x 1mm x 1mm
    resampled_volume_A = resample_volume(arr2, 
                                         [os[0], os[1], (max(zaxis) - min(zaxis)) / len(zaxis)], 
                                         target_spacing)
    p, s = pth.split('/')[-3], pth.split('/')[-2]
    np.save(rescaled_path + f'/{p}_{s}', (resampled_volume_A * 255.0).astype(np.uint8))
    
    
p = Pool(processes=6)
print(len(glob.glob(path + f'/../input/train_images/*/*/')))
for pth in tqdm(glob.glob(path + f'/../input/train_images/*/*/')):
    p.apply_async(process, (pth, ))
p.close()
p.join()

todo = [x for x in glob.glob(rescaled_path + '/*.npy')]# if not x.split('/')[-1].split('.')[0] in exist]

new_roi = {}
shape_stats = {}
for pth in tqdm(todo):
    resampled_volume_A = np.load(pth)

    ipt = [cv2.resize((resampled_volume_A[:, :, i]).astype(np.uint8), (512, 512)
                 ) for i in range(resampled_volume_A.shape[2])]

    ipt = [tensor_tfms(np.stack([e]*3, -1)) for e in ipt]

    ipt = torch.stack(ipt)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            r = model(ipt.cuda())
            r2 = four_model(ipt.cuda())
        r = torch.sigmoid(r.float())
        r2 = torch.sigmoid(r2.float())
        r = torch.nn.functional.interpolate(
            r, size=(resampled_volume_A.shape[0], resampled_volume_A.shape[1])
        ).cpu().numpy()
        r2 = torch.nn.functional.interpolate(
            r2, size=(resampled_volume_A.shape[0], resampled_volume_A.shape[1])
        ).cpu().numpy()

        
    r2 = r2 > 0.5
    r = r > 0.5
    res = np.concatenate([r2, r], 1)
    
    xr = np.where(res[:, :4, :, :].reshape(res.shape[0], -1).sum(1) > 0)[0]
    xmin, xmax = xr.min(), xr.max()

    
    if xmax - xmin < 12:
        new_roi[pth.split('/')[-1].split('.')[0]] = [0, int(res.shape[0])]
    else:
        new_roi[pth.split('/')[-1].split('.')[0]] = [int(xmin), int(xmax)]
        
    sh = res.shape
    packed_data = np.packbits(res)
    np.save(pth.replace('rescaled', 'masks'), packed_data)
    shape_stats[pth.split('/')[-1]] = sh
    
with open(path + '/../input/cache_100/shape_state.pkl', 'wb') as fp:
    pk.dump(shape_stats, fp)
    
with open(path + '/../input/cache_100/zroi_new.json', 'w') as fp:
    json.dump(new_roi, fp)