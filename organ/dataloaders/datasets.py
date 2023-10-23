import numpy as np
import os
import pandas as pd
from skimage.io import imread
from pathlib import Path
import cv2
import json
import pickle
from typing import Callable, List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import skimage
# from utils.tile_fix import tile
# from utils.ha import get_tiles
import random
from configs import Config
import tifffile
import torch
import math
import glob
import ast
import torchvision
import albumentations


def normwidth(size, margin=32):
    outsize = size // margin * margin
    outsize = max(outsize, margin)
    return outsize


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(math.ceil(img.shape[1] * percent))
    resized_height = int(math.ceil(img.shape[0] * percent))

    # resized_width = normwidth(resized_width)
    # resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized


class TrainDataset(Dataset):
    HEIGHT = 137
    WIDTH = 236

    def __init__(self, df: pd.DataFrame, images: pd.DataFrame,
                 image_transform: Callable, debug: bool = True, weighted_sample: bool = False,
                 square: bool = False):
        super().__init__()
        self._df = df
        self._images = images
        self._image_transform = image_transform
        self._debug = debug
        self._square = square
        # stats = ([0.0692], [0.2051])
        self._tensor_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if weighted_sample:
            # TODO: if weight sampler is necessary
            self.weight = self.get_weight()

    def get_weight(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/../../metadata/train_onehot.pkl'
        onehot = pd.read_pickle(path)
        exist = onehot.loc[self._df['id']]
        weight = []
        log = 1 / np.log2(exist.sum() + 32)
        for i in range(exist.shape[0]):
            weight.append((log * exist.iloc[i]).max())
        weight = np.array(weight)
        return weight

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = self._images[idx].copy()
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # image = Image.fromarray(image)
        if self._image_transform:
            image = self._image_transform(image=image)['image']
        # else:
        image = self._tensor_transform(image)
        target = np.zeros(3)
        target[0] = item['grapheme_root']
        target[1] = item['vowel_diacritic']
        target[2] = item['consonant_diacritic']
        return image, target


class LandmarkDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, full=False):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tta = tta
        self.test = test
        self.full = full

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        image_id = self.df.iloc[idx % self.df.shape[0]]['id']
        prefix = 'full' if self.full else 'clean_data'
        path = self.path / '../../../landmark/{}/train/{}/{}/{}/{}.jpg'.format(prefix,
            image_id[0], image_id[1], image_id[2], image_id
        )
        # img = imread(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        # resize short edge first
        # img = resize_short(img, self.cfg.transform.size)

        if self.tfms:
            return self.tensor_tfms(self.tfms(image=img)['image']), self.df.iloc[idx % self.df.shape[0]]['label']
        else:
            return self.tensor_tfms(img), self.df.iloc[idx % self.df.shape[0]]['label']


class STRDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, prefix='train'):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tta = tta
        self.test = test
        # self.prefix = prefix
        # self.seq_cvt = {x.split('_')[-1].split('.')[0]: x.split('/')[-1]
        #                 for x in glob.glob(str(self.path / '../../input/{}/*/*/*.jpg'.format(self.prefix)))}
        #

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx % self.df.shape[0]]
        path = str(self.path / '../../input/train_images/{}'.format(
            item['image_id']
        ))
        # print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.cfg.transform.size == 512:
            # indeed size are 800 x 600
            # however, a error set default as 512
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        if self.tfms:
            return (
                self.tensor_tfms(self.tfms(image=img)['image']),
                item.label
            )
        else:
            return (
                self.tensor_tfms(img),
                item.label
            )


class RANZERDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        target_cols = self.df.iloc[:, 1:12].columns.tolist()
        self.labels = self.df[target_cols].values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        if (self.cfg.transform.size == 256 or self.cfg.transform.size == 300 or self.cfg.transform.size == 512) and os.path.exists(self.path / '../../input/train512'):
            path = str(self.path / '../../input/train512/{}.jpg'.format(
                row.StudyInstanceUID
            ))
        # elif self.cfg.transform.size == 384:
        #     path = str(self.path / '../../input/train384/{}.jpg'.format(
        #         row.StudyInstanceUID
        #     ))
        else:
            path = str(self.path / '../../input/train/{}.jpg'.format(
                row.StudyInstanceUID
            ))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        if not img.shape[0] == self.cfg.transform.size:
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))

        # img = img.astype(np.float32)
        # img = img.transpose(2, 0, 1)
        label = torch.tensor(self.labels[index]).float()
        img = self.tensor_tfms(img)
        if self.mode == 'test':
            return img
        else:
            return img, label
        # if self.mode == 'test':
        #     return torch.tensor(img).float()
        # else:
        #     return torch.tensor(img).float(), label



COLOR_MAP = {'ETT - Abnormal': (255, 0, 0),
             'ETT - Borderline': (0, 255, 0),
             'ETT - Normal': (0, 0, 255),
             'NGT - Abnormal': (255, 255, 0),
             'NGT - Borderline': (255, 0, 255),
             'NGT - Incompletely Imaged': (0, 255, 255),
             'NGT - Normal': (128, 0, 0),
             'CVC - Abnormal': (0, 128, 0),
             'CVC - Borderline': (0, 0, 128),
             'CVC - Normal': (128, 128, 0),
             'Swan Ganz Catheter Present': (128, 0, 128),
            }


class AnnotedDataset(Dataset):
    def __init__(self, df, df_annotations, annot_size=50, transform=None, cfg=None):
        self.df = df
        self.df_annotations = df_annotations
        self.annot_size = annot_size
        self.file_names = df['StudyInstanceUID'].values
        target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']
        self.labels = df[target_cols].values
        self.transform = transform
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cfg = cfg
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # self.path / '../../input/train512/{}.jpg
        file_path = str(self.path / f'../../input/train512_png/{file_name}.png')
        image_raw = cv2.imread(file_path)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # image_raw = image.copy()
        if file_name in self.df_annotations.StudyInstanceUID.unique():
            is_annotated = True
            file_path = str(self.path / f'../../input/train_anno_png/{file_name}.png')
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            is_annotated = False
            image = image_raw.copy()
        if self.transform:
            if not self.cfg.experiment.unified_tfms:
                augmented = self.transform(image=image)
                image = augmented['image']
                if is_annotated:
                    augmented_raw = self.transform(image=image_raw)
                    image_raw = augmented_raw['image']
                else:
                    image_raw = image.copy()
            else:
                if is_annotated:
                    tf = self.transform(image=image, image1=image_raw)
                    image_raw = tf['image1']
                    image = tf['image']
                else:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                    image_raw = image.copy()
        if not image.shape[0] == self.cfg.transform.size:
            image = cv2.resize(image, (self.cfg.transform.size, self.cfg.transform.size))
        if not image_raw.shape[0] == self.cfg.transform.size:
            image_raw = cv2.resize(image_raw, (self.cfg.transform.size, self.cfg.transform.size))
        image = self.tensor_tfms(image)
        image_raw = self.tensor_tfms(image_raw)
        label = torch.tensor(self.labels[idx]).float()
        return image, image_raw, is_annotated, label


class RANZCRSegDataset(Dataset):
    def __init__(self, df, cfg=None, tfms=None):
        self.df = df
        self.cfg = cfg
        self.tfms = tfms
        self.cols = ['class{}'.format(i) for i in range(19)]

        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.406]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.com = albumentations.JpegCompression(quality_lower=90, quality_upper=90, p=1)
        self.col_sums = self.df.set_index('ID')[self.cols].sum(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.df.iloc[idx]
        if not s.is_external:
            if self.cfg.data.image_prefix:
                img_prefix = self.cfg.data.image_prefix
            elif self.cfg.transform.size <= 512:
                img_prefix = 'train512_png'
            elif self.cfg.transform.size == 768 and os.path.exists(self.path / f'../../input/train768_png'):
                img_prefix = 'train768_png'
            elif self.cfg.transform.size == 1024 and os.path.exists(self.path / f'../../input/train1024_png'):
                img_prefix = 'train1024_png'
            else:
                img_prefix = 'train'
            # print(f'[ ! ] Use predefined mask: {seg_prefix}, and img: {img_prefix}')
            # print(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.jpg')
            if 'jpeg' in img_prefix:
                r = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.jpg'), 0)
                g = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_green.jpg'), 0)
                b = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_blue.jpg'), 0)
                a = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_yellow.jpg'), 0)
            else:
                r = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.png'), 0)
                g = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_green.png'), 0)
                b = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_blue.png'), 0)
                a = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_yellow.png'), 0)
            img = np.stack([r, g, b, a], -1)
            # img = np.stack([r, g, b], -1)
            # print(img.shape)
        else:
            # print(s.img_path)
            if 'jpg' in s.img_path:
                # print(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.jpg'))
                r = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.jpg'), 0)
                g = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_green.jpg'), 0)
                b = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_blue.jpg'), 0)
                a = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_yellow.jpg'), 0)
            else:
                r = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.png'), 0)
                g = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_green.png'), 0)
                b = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_blue.png'), 0)
                a = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_yellow.png'), 0)
            try:
                img = np.stack([r, g, b, a], -1)
            except:
                print(f'fuck: {s.ID}')
                img = np.zeros((512, 512, 4))
        if self.tfms:
            # print(s)
            tf = self.tfms(image=img)
            img = tf['image']
        if not img.shape[0] == self.cfg.transform.size:
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        img = self.tensor_tfms(img)
        # print(type(img), type(seg), type(s[self.cols].values))
        return img, torch.tensor(s[self.cols].values.astype(np.float))


class COVIDDataset(Dataset):
    def __init__(self, df, cfg=None, tfms=None):
        self.df = df
        self.cfg = cfg
        self.tfms = tfms
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
#        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.path = Path(os.path.dirname(os.path.realpath(__file__))) / '../'

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        res = np.load(
            self.path / '../input/contrails-images-ash-color/contrails/{}.npy'.format(row.record_id)).astype(np.float32)
        img, mask = res[:, :, :3], res[:, :, 3]
        original_mask = mask.copy()
        # print(img.dtype, mask.dtype)
        if self.tfms:
            tf = self.tfms(image=img, mask=mask)
            img = tf['image']
            mask = tf['mask'] > 0
        if not img.shape[0] == self.cfg.transform.size or not img.shape[1] == self.cfg.transform.size:
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))

        # if not mask.shape[0] == self.cfg.transform.size or not mask.shape[1] == self.cfg.transform.size:
        #     mask = cv2.resize((mask*255.0).astype(np.uint8), (self.cfg.transform.size, self.cfg.transform.size)) > 128

        img = self.tensor_tfms(img)
        return img, mask.reshape(1, mask.shape[0], mask.shape[1]), original_mask


def get_label(row):
    return (
        row.bowel_injury,
        row.extravasation_injury,
        row[['kidney_healthy', 'kidney_low', 'kidney_high']].values.argmax(),
        row[['liver_healthy', 'liver_low', 'liver_high']].values.argmax(),
        row[['spleen_healthy', 'spleen_low', 'spleen_high']].values.argmax(),
        row.any_injury
    )


class CustomPNGSequenceDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):
        self.df = df
        self.transform = tfms
        self.cfg = cfg
        self.mode = mode
        self.N = self.cfg.data.N
        if cfg.model.name in []: #['tu-maxvit_tiny_tf_384_unete']:
            self.tensor_tfms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
            ])
        else:
            self.tensor_tfms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__))) / '../..'

        self.zroi = json.load(open(self.path / 'input/cache_100/zroi_new.json'))
        self.shape_dict = pickle.load(open(self.path / 'input/cache_100/shape_state.pkl', 'rb'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = np.load(self.path / f'input/cache_100/rescaled/{row.patient}_{row.series}.npy')
        # our channel
        # other_mask = np.load(f'/media/rd0/rsna_cache/rescaled_high_mask/{row.patient}_{row.series}.npy')
        path = self.path / f'input/cache_100/masks/{row.patient}_{row.series}.npy'
        other_mask_loaded = np.load(path)
        path = str(path).split('/')[-1]
        other_mask_flat = np.unpackbits(other_mask_loaded)[:np.product(self.shape_dict[path])]
        other_mask_raw = other_mask_flat.reshape(self.shape_dict[path])
        other_mask = other_mask_raw[:, :4, :, :]
        body_mask = other_mask_raw[:, 4, :, :]

        left, right = self.zroi[f'{row.patient}_{row.series}']
        raw = raw[:, :, left:right]
        body_mask = body_mask[left:right]
        other_mask = other_mask[left:right]

        w, h, d = raw.shape
        N = self.N

        slices = []
        masks = []
        gap = int(np.floor(d / N))
        offset = d - gap * (N - 1)
        if self.mode == 'train':
            start = np.random.randint(0, offset)
        else:
            start = offset // 2
        for i in range(N):
            mask = body_mask[start + gap * i]
            croped = raw[:, :, start + gap * i] * mask
            x, y = np.where(mask > 0)
            slices.append(croped[x.min():x.max(), y.min():y.max()])
            masks.append((other_mask[start + gap * i, :, x.min():x.max(), y.min():y.max()] * 255.0).astype(np.uint8))

        # for m in masks:
        #     print(m.shape)

        slices_new = []
        masks_new = []
        if self.transform:
            for i, e in enumerate(slices):
                tfs = self.transform(image=e, mask=np.rollaxis(masks[i], 0, 3))
                slices_new.append(tfs['image'])
                masks_new.append(np.rollaxis(tfs['mask'] > 128, 2, 0))
                # masks_new.append(tfs['mask'] > 128)
        else:
            # just resize
            slices_new = [cv2.resize(s, (self.cfg.transform.size, self.cfg.transform.size)) for s in slices]
            masks_new = [cv2.resize(m, (self.cfg.transform.size, self.cfg.transform.size)) > 128 for m in masks]
        slices = [self.tensor_tfms(np.stack([e] * 3, -1)) for e in slices_new]

        vol = np.stack(slices, 0).astype(np.float32)
        a, b, c, d, e, f = get_label(row)
        # print(row.iloc[4:-1].values)
        return vol, a, b, c, d, e, f, np.stack(masks_new), row.iloc[4:-2].values.astype(int)
