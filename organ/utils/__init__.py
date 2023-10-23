import argparse
import os
import torch
import numpy as np
import random
import requests as req
import json
from configs import get_config, Config
from path import Path
import pandas as pd
# import neptune
from torch.utils.data import Sampler, Dataset, DistributedSampler
import torch.distributed as dist
import math
from operator import itemgetter


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas = None,
        rank = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # if self.shuffle:
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # else:
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


GRAPHEME_ROOT = 168 #grapheme_root: 168
VOWEL_DIACRITIC = 11 #vowel_diacritic: 11
CONSONANT_DIACRITIC = 7 #consonant_diacritic: 7

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ


def idoit_collect_func(batch):
    img, study, lbl, image_as_study, bbox = [], [], [], [], []
    for im, st, lb, ias, bb in batch:
        img.extend(im)
        study.extend(st)
        lbl.append(lb)
        image_as_study.extend(ias)
        bbox.append(bb)
    return torch.stack(img), study, torch.tensor(lbl), torch.tensor(image_as_study), bbox


def tile(img, sz=128, N=12):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1-pad1//2], [0, 0]],
                 constant_values=255)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        img = np.pad(img, [[0, N-len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    return img


def load_matched_state(model, state_dict):
    model_dict = model.state_dict()
    not_loaded = []
    for k, v in state_dict.items():
        if k in model_dict.keys():
            if not v.shape == model_dict[k].shape:
                print('Error Shape: {}, skip!'.format(k))
                continue
            model_dict.update({k: v})
        else:
            # print('not matched: {}'.format(k))
            not_loaded.append(k)
    if len(not_loaded) == 0:
        print('[ √ ] All layers are loaded')
    else:
        print('[ ! ] {} layer are not loaded'.format(len(not_loaded)))
    model.load_state_dict(model_dict)


def load_matched_state_ddp(model, state_dict):
    model_dict = model.state_dict()
    not_loaded = []
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if k in model_dict.keys():
            if not v.shape == model_dict[k].shape:
                print('Error Shape: {}, skip!'.format(k))
                continue
            model_dict.update({k: v})
        else:
            # print('not matched: {}'.format(k))
            not_loaded.append(k)
    if len(not_loaded) == 0:
        print('[ √ ] All layers are loaded')
    else:
        print('[ ! ] {} layer are not loaded'.format(len(not_loaded)))
    model.load_state_dict(model_dict)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def prepare_for_result(cfg: Config):
    # print(cfg.train.dir)
    if not os.path.exists(cfg.train.dir):
        raise Exception('Result dir not found')
    if os.path.exists(cfg.train.dir + '/' + cfg.basic.id):
        if cfg.basic.debug:
            print('[ X ] The output dir already exist!')
            output_path = Path(cfg.train.dir) / cfg.basic.id
            return output_path
        else:
            raise Exception('The output dir already exist')
    output_path = Path(cfg.train.dir) / cfg.basic.id
    os.mkdir(output_path)
    os.mkdir(output_path / 'checkpoints')
    os.mkdir(output_path / 'logs')
    with open(output_path / 'train.log', 'w') as fp:
        fp.write(
            'Fold\tEpochs\tlr\ttrain_loss\tvalid_loss\tdice\tbowel\tkidney\textravasation\tliver\tspleen\tany\tscore\n'
        )
    return output_path


def parse_args(mode='sp'):
    '''
    Whether call this function in CLI or docker utility, we parse args here
    For inference, we use 2 worker
    GPU, run_id,

    :return:
    '''
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test', 'valid'])
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--debug', type=bool, default=False)
    arg('--search', type=bool, default=False)
    arg('--gpu', type=str, default='0')
    arg('-i', type=str, default='')
    arg('-j', type=str, default='')
    arg('--seed', type=int, default=-1)
    arg('-f', type=str, default='')
    arg('-s', type=str, default='')
    arg('--tta', type=int, default=4)
    arg('--tta_tfms', type=str, default='none')
    # temp argument, absolute path
    arg('--checkpoint', type=str, default='')
    # temp argument, runfold
    arg('--run_fold', type=int, default=0)
    arg('-b', type=int, default=1)
    arg('--select', type=str, default='valid_loss')
    arg('--epoch', type=int, default=-1)
    arg('--tag', type=str, default='')

    args = parser.parse_args()

    # load the configs
    if 'http' in args.j:
        raise Exception('Not implemented Yet')
    elif args.mode == 'validate':
        path = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / '../results/' / args.i / 'config.json'
        cfg = Config.load(path)
    else:
        cfg = get_config(args.j)

    if not args.seed == -1:
        cfg.basic.seed = args.seed
    # Initial jobs
    # set seed, should always be done
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)

    # set the gpu to use
    print('[ √ ] Using #{} GPU'.format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join(args.gpu))
    # print(','.join(args.gpu))

    cfg.basic.GPU = args.gpu
    cfg.basic.debug = args.debug
    cfg.basic.search = args.search
    cfg.basic.id = args.i
    cfg.basic.mode = args.mode

    # print(args.b)
    cfg.dpp.sb = args.b == 1
    cfg.dpp.checkpoint = args.checkpoint
    cfg.dpp.mode = args.mode
    # print(args.mode)

    # if mode == 'sp' and not args.debug:
    #     neptune.init(project_qualified_name='sheep/kaggle-ranzcr',
    #                  api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZGM2NWQ4MWEtNTEwMS00YjQxLTk2ZTItMzAwMTdiNDU5Y2Q5In0='
    #                  )  # add your credentials
    #     neptune.create_experiment(args.i, tags=[
    #         'fold{}'.format(cfg.experiment.run_fold),
    #         cfg.model.name, args.i, cfg.transform.name, cfg.transform.size,
    #         cfg.loss.name, args.tag or 'not assigned'
    #     ], params=cfg.to_flatten_dict())


    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return args, cfg


def load_search(json_path, seed=233):
    '''
    Whether call this function in CLI or docker utility, we parse args here
    For inference, we use 2 worker
    GPU, run_id,

    :return:
    '''
    # parser = argparse.ArgumentParser()
    # arg = parser.add_argument
    # arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test', 'valid'])
    # arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    # arg('--debug', type=bool, default=False)
    # arg('--search', type=bool, default=False)
    # arg('--gpu', type=str, default='0')
    # arg('-i', type=str, default='')
    # arg('-j', type=str, default='')
    # arg('--seed', type=int, default=-1)
    # arg('-f', type=str, default='')
    # arg('-s', type=str, default='')
    # arg('--tta', type=int, default=2)
    # # temp argument, absolute path
    # arg('--checkpoint', type=str, default='')
    # # temp argument, runfold
    # arg('--run_fold', type=int, default=0)
    # arg('-b', type=int, default=1)
    #
    # args = parser.parse_args()

    # load the configs
    cfg = get_config(json_path)

    cfg.basic.seed = seed
    # Initial jobs
    # set seed, should always be done
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)

    # set the gpu to use
    # print('[ √ ] Using #{} GPU'.format(args.gpu))
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join(args.gpu))
    # print(','.join(args.gpu))

    # cfg.basic.GPU = args.gpu
    # cfg.basic.debug = args.debug
    # cfg.basic.search = args.search
    # cfg.basic.id = args.i
    # cfg.basic.mode = args.mode
    #
    # # print(args.b)
    # cfg.dpp.sb = args.b == 1
    # cfg.dpp.checkpoint = args.checkpoint
    # cfg.dpp.mode = args.mode
    # print(args.mode)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    return cfg


def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

