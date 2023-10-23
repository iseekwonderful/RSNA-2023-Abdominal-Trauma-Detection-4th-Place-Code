from utils import Config, idoit_collect_func
import pandas as pd
from path import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataloaders.datasets import STRDataset, LandmarkDataset, RANZERDataset, AnnotedDataset, CustomPNGSequenceDataset
# from dataloaders.transform import get_tfms
from dataloaders.transform_loader import get_tfms
from collections import Counter, defaultdict
import pickle
import os
import numpy as np
from dataloaders.sampler import RandomBatchSampler
import random
import albumentations as A
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop, RandomResizedCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import json
import uuid


class AnnotatedRandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        # load configs

        train = pd.read_csv(path / 'split' / cfg.data.csv)#.drop('postfix', axis=1)

        self.full_df = train.copy()

        train['is_external'] = False
        # train['img_path'] = ''
        # train['msk_path'] = ''

        # self.annotation = pd.read_csv(path / 'split' / 'train_annotations.csv')

        self.train_meta, self.valid_meta = (train[train.fold != cfg.experiment.run_fold],
                                            train[train.fold == cfg.experiment.run_fold])

        if self.cfg.data.filter_empty:
            print('[!] filter empty!')
            self.train_meta = self.train_meta[self.train_meta.area > 0].copy()

        if cfg.data.diff > 0:
            print(f'[ ! ] We denoise based on oofs, before: {self.train_meta.shape[0]}')
            self.train_meta = self.train_meta[self.train_meta['diff'] < cfg.data.diff]
            print(f'[ ! ] We denoise based on oofs, after: {self.train_meta.shape[0]}')
        # if len(self.cfg.experiment.externals) > 0:
        #     for i, x in enumerate(self.cfg.experiment.externals):
        #         print(f'[ i ] Load external dataset: {x.format(cfg.experiment.run_fold)}')
        #         data = pd.read_csv(path / 'split' / x)
        #         data['is_external'] = True
        #         data = data[~data.ID.isin(['406_G4_1', '802_C3_1', '92_F8_1', '1615_A8_4', '140_F12_1'])]
        #         data['img_path'] = self.cfg.data.external_image_prefixs[i]
        #         print('[ i ] Image looks like: {}'.format(data['img_path'][0]))
        #         self.train_meta = pd.concat([self.train_meta, data])
        #
        # if self.cfg.experiment.anno_only and self.cfg.experiment.keep_valid:
        #     b = self.train_meta.shape[0]
        #     self.train_meta = self.train_meta[self.train_meta.StudyInstanceUID.isin(self.annotation.StudyInstanceUID.unique())]
        #
        #     print(f'[ W ] Annotation Only mode keep valid, before: {b}, after: {self.train_meta.shape[0]}')

        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.05)
            self.valid_meta = self.valid_meta.sample(frac=0.3)

    def get_dataloader(self, test_only=False, train_shuffle=True, infer=False, tta=-1, tta_tfms=None):
        if test_only:
            raise NotImplementedError()
            # if tta == -1:
            #     tta = 1
            # path = Path(os.path.dirname(os.path.realpath(__file__)))
            # test = pd.read_csv(path / '../../input/kaggle/sample_submission.csv')
            # test['prefix'] = 'kaggle'
            # test_ds = SIIMDataset(test, tta_tfms, size=self.cfg.transform.size, tta=tta, test=True)
            # test_dl = DataLoader(dataset=test_ds, batch_size=self.cfg.eval.batch_size,
            #                       num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
            # return test_dl
        print('[ √ ] Using transformation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)
        # augmentation end
        # def __init__(self, df, tfms=None, cfg=None, mode='train'):
        # train_ds = STRDataset(self.train_meta, train_tfms, size=self.cfg.transform.size,
        #                       cfg=self.cfg, prefix=self.cfg.experiment.preprocess)
        # train_ds = RANZERDataset(df=self.train_meta, tfms=train_tfms, cfg=self.cfg, mode='train')
        train_ds = CustomPNGSequenceDataset(self.train_meta, tfms=train_tfms, cfg=self.cfg)
        if self.cfg.experiment.weight and train_shuffle:
            train = self.train_meta.copy()
            method_dict = {
                'sqrt': np.sqrt,
                'log2': np.log2,
                'log1p': np.log1p,
                'log10': np.log10,
                'as_it_is': lambda w: w
            }
            if self.cfg.experiment.method in method_dict:
                print('[ √ ] Use weighted sampler, method: {}'.format(self.cfg.experiment.method))
                weight = (1 / method_dict[self.cfg.experiment.method](
                    train[['cancer', 'fold']].groupby('cancer').transform('count')['fold'])).values
                print(weight)
                # weight = 1 / method(train.groupby('label').transform('count').image_id.values)
            elif 'pow' in self.cfg.experiment.method:
                p = float(self.cfg.experiment.method.replace('pow_', ''))
                print('[ √ ] Use weighted sampler, method: Power of {}'.format(p))
                for x in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
                    train['{}_p'.format(x)] = (1 / np.power(
                        train[[x, 'fold']].groupby(x).transform('count')['fold'].values, p)
                                               ) / len(train[x].value_counts())
                weight = train[['grapheme_root_p', 'vowel_diacritic_p', 'consonant_diacritic_p']].max(1).values
            else:
                raise Exception('Unknown weighting method!')
            rs = WeightedRandomSampler(weights=weight, num_samples=len(weight))
            train_dl = DataLoader(train_ds, sampler=rs, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        elif self.cfg.experiment.batch_sampler:
            print('[ i ] Batch Sampler!')
            bs = RandomBatchSampler(train_ds.df, self.cfg.train.batch_size, cfg=self.cfg)
            train_dl = DataLoader(dataset=train_ds, batch_sampler=bs, collate_fn=idoit_collect_func,
                                  num_workers=self.cfg.transform.num_preprocessor)
        else:
            train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor, #collate_fn=idoit_collect_func,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)
        if tta == -1:
            tta = 1
        # valid_ds = STRDataset(self.valid_meta, val_tfms, size=self.cfg.transform.size, tta=tta,
        #                       cfg=self.cfg, prefix=self.cfg.experiment.preprocess)
        # def __init__(self, df, tfms=None, cfg=None, mode='train'):
        # valid_ds = RANZERDataset(df=self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='train')
        valid_ds = CustomPNGSequenceDataset(self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='valid')
        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size, #collate_fn=idoit_collect_func,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        return train_dl, valid_dl, None, self.full_df
