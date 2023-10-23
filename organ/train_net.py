import cv2
cv2.setNumThreads(0)

import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss, get_class_balanced_weighted, get_log_weight
from losses.regular import class_balanced_ce
from optimizers import get_optimizer
from dist_train import basic_train, basic_validate
from scheduler import get_scheduler
from utils import load_matched_state, load_matched_state_ddp
from configs import Config
from torch.utils.tensorboard import SummaryWriter
import torch
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
# from apex.parallel import convert_syncbn_model, SyncBatchNorm
from torch.nn import SyncBatchNorm
from pathlib import Path
import random
from transformers import get_linear_schedule_with_warmup, AdamW
from collections import OrderedDict
from utils import DistributedWeightedSampler, DistributedSamplerWrapper
import timm


from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm

import warnings
warnings.filterwarnings('ignore')


def train(gpu, cfg: Config):
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.cuda.set_device(gpu)
    if gpu == 0 and not cfg.dpp.mode == 'valid':
        result_path = prepare_for_result(cfg)
        writer = SummaryWriter(log_dir=result_path)
        cfg.dump_json(result_path / 'config.json')
    elif cfg.dpp.mode == 'valid':
        result_path = Path(cfg.train.dir) / cfg.basic.id
        mode, ckp = cfg.dpp.mode, cfg.dpp.checkpoint
        cfg = Config.load(result_path / 'config.json')
        cfg.dpp.mode, cfg.dpp.checkpoint = mode, ckp
    else:
        result_path = None
        writer = None
    # init basic elements
    rank = cfg.dpp.rank * cfg.dpp.nodes + gpu
    word_size = cfg.dpp.gpus * cfg.dpp.nodes
    print(word_size)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=cfg.dpp.gpus * cfg.dpp.nodes,
        rank=rank
    )
    if cfg.experiment.run_fold == -1:
        range_length = cfg.experiment.fold
    else:
        range_length = 1
    for idx in range(range_length):
        fold_idx = cfg.experiment.run_fold if range_length == 1 else idx
        if range_length > 1:
            cfg.experiment.run_fold = fold_idx
        if gpu == 0:
            print(f'[ ! ] Start Training Fold: {cfg.experiment.run_fold}')
        exp = get_dataloader(cfg)(cfg)
        train_dl, valid_dl, test_dl, full_df = exp.get_dataloader()
        # should not do like this - -, maybe we can, since works fine
        train_ds, valid_ds = train_dl.dataset, valid_dl.dataset
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=word_size, rank=rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_ds, num_replicas=word_size, rank=rank)
        train_sampler = DistributedSamplerWrapper(train_dl.sampler, num_replicas=word_size, rank=rank)
        # valid_sampler = DistributedSamplerWrapper(valid_dl.sampler, num_replicas=word_size, rank=rank)
        train_dl = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.transform.num_preprocessor,
            pin_memory=True,
            sampler=train_sampler)
        valid_dl = torch.utils.data.DataLoader(
            dataset=valid_ds,
            batch_size=cfg.train.batch_size // 2,
            shuffle=False,
            num_workers=cfg.transform.num_preprocessor,
            pin_memory=False,
            sampler=valid_sampler)
        if gpu == 0:
            print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
        model = get_model(cfg).cuda(gpu)
        # if necessary load checkpoint
        if cfg.dpp.mode == 'train':
            if cfg.model.from_checkpoint and not cfg.model.from_checkpoint == 'none':
                # from path import Path
                # import os
                # path = Path(os.path.dirname(os.path.realpath(__file__))) / '../input/'
                # print(path, cfg.model.from_checkpoint)
                checkpoint = cfg.model.from_checkpoint
                if gpu == 0:
                    print('[ ! ] Loading checkpoint from {}.'.format(checkpoint))
                load_matched_state_ddp(model, torch.load(checkpoint, map_location='cpu'))
        elif cfg.dpp.mode == 'valid':
            if not cfg.dpp.checkpoint:
                raise Exception('Validation please provide a path')
            print('[ ! ] Loading checkpoint from {}.'.format(cfg.dpp.checkpoint))
            load_matched_state(model, torch.load(cfg.dpp.checkpoint,
                                                 map_location={'cuda:0': 'cuda:{}'.format(gpu)}))
        if cfg.loss.name == 'weight_ce':
            # if we use weighted ce loss, we load the loss here.
            weights = torch.Tensor(cfg.loss.param['weight']).cuda()
            print('[ ! ] Weight CE!, {}'.format(weights.cpu().numpy()))
            loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
        elif cfg.loss.weight_type == 'log1p':
            weights = (1 / np.log1p(full_df['id_label'].value_counts())).sort_values().sort_index().values
            cfg.loss.param['weight'] = weights
            loss_func = get_loss(cfg)
        elif cfg.loss.weight_type == 'sqrt':
            weights = (1 / np.sqrt(full_df['id_label'].value_counts())).sort_values().sort_index().values
            cfg.loss.param['weight'] = weights
            loss_func = get_loss(cfg)
        else:
            loss_func = get_loss(cfg)
        optimizer = get_optimizer(model, cfg)
        if gpu == 0:
            print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
        # if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
        #     model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
        if not cfg.scheduler.name == 'none':
            scheduler = get_scheduler(cfg, optimizer, len(train_dl))
        else:
            scheduler = None
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # num_train_optimization_steps = int(cfg.train.num_epochs * len(train_dl) / 1)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.optimizer.param['lr'],
        #                   correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
        #                                             num_training_steps=num_train_optimization_steps)  # PyTorch schedule
        # fixme
        if cfg.dpp.sb:
            # pass
            print('[ ! ] Convert to SYNCBN')
            if 'tu-' in cfg.model.name or 'timm' in cfg.model.name:
                model = convert_sync_batchnorm(model).cuda(gpu)
            else:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[gpu], find_unused_parameters=True)
        if cfg.dpp.mode == 'train':
            #def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, gpu, tune=None):
            basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer, gpu=gpu)
        elif cfg.dpp.mode == 'valid':
            basic_validate(model, valid_dl, loss_func, cfg, gpu)
        else:
            raise Exception('Unknown mode!')


if __name__ == '__main__':
    args, cfg = parse_args(mode='mp')
    cfg.dpp.gpus = len(cfg.basic.GPU)
    # print(cfg.dpp.sb)
    if not cfg.dpp.sb:
        print('[ x ] DPP without SyncBN warning!')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8877'
    print(cfg.dpp.gpus)
    print(cfg)
    mp.spawn(train, nprocs=cfg.dpp.gpus, args=(cfg,))