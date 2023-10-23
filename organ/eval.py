import numpy as np
import pandas as pd
import tqdm
import cv2
import glob
import math
import torch
import operator
import sys
import os
from path import Path
from torch import nn

from shutil import copyfile

from utils import parse_args, prepare_for_result
from torch.utils.data import DataLoader, Dataset
from models import get_model
from losses import get_loss, get_class_balanced_weighted
from dataloaders import get_dataloader
from utils import load_matched_state
from configs import Config
import seaborn as sns
from dataloaders.transform_loader import get_tfms
import torchvision
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from scipy.stats import pearsonr
# import plotly.express as pe
from skimage.io import imread
sns.set()
import  argparse


def oof_eval_run(run_id):
    fold2models = {i: [] for i in range(5)}
    df = pd.read_csv(f'../results/{run_id}/train.log', sep='\t').reset_index()
    fold2epochs = {}
    fold_predicted = []
    all_preds = []
    all_truths = []
    oofs = []

    stats = {0.15: [], 0.3: [], 0.45: [], 0.6: []}

    for f in range(5):
        # valid_accuracy
        # acc@.50
        eph = df[df['index'] == f].sort_values('valid_loss', ascending=False).iloc[0].Fold
        fold2epochs[f] = int(eph)
        cfg_path = f'../results/{run_id}/config.json'
        mdl_path = glob.glob(f'../results/{run_id}/checkpoints/f{f}*-{fold2epochs[f]}*')[0]
        # copyfile(mdl_path, './sub_v0_models/' + mdl_path.split('/')[-3] + '_' + mdl_path.split('/')[-1])
        torch.cuda.empty_cache()
        cfg = Config.load_json(cfg_path)
        cfg.train.batch_size = 32
        cfg.experiment.run_fold = f
        train_dl, valid_dl, test_dl, _ = get_dataloader(cfg)(cfg).get_dataloader()
        # model = EfficinetNet().cuda()
        model = get_model(cfg).cuda()
        load_matched_state(model, torch.load(mdl_path))
        loss_func = get_loss(cfg)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _ = model.eval()

        with torch.no_grad():
            results = []
            sop_ids = []
            original_truth = []
            losses, predicted, predicted_p, truth = [], [], [], []
            for i, (img, lbl, original_mask) in enumerate(tqdm.tqdm(valid_dl)):
                img, lbl, original_mask = img.cuda(), lbl.cuda(), original_mask.float()
                with torch.cuda.amp.autocast():
                    logits = model(img).float()
                    # logits1 = model(img.flip(-1)).float()
                    # logits2 = model(img.flip(-2)).float()
                logits = torch.sigmoid(logits.float().cpu())
                # logits1 = torch.sigmoid(logits1.float().cpu()).flip(-1)
                # logits2 = torch.softmax(logits2.cpu(), 1)
                cls_pred = logits.numpy()
                # cls_pred = ((logits1 + logits) / 2).numpy()
                predicted_p.append(cls_pred)
                truth.append(lbl.cpu().numpy())
                original_truth.append(original_mask.numpy())

            predicted_p = np.concatenate(predicted_p)
            truth = np.concatenate(truth)
            original_truth = np.concatenate(original_truth)
            if not predicted_p.shape[3] == 256:
                predicted_p = torch.nn.functional.interpolate(torch.tensor(predicted_p), size=(256, 256)).numpy()
            predicted_p = predicted_p.reshape(-1, 256, 256)
            for thr in [0.15, 0.3, 0.45, 0.6]:
                d = 2 * ((predicted_p > thr) * original_truth).sum() / (
                            (predicted_p > thr) + original_truth).sum()
                stats[thr].append(d)
                # print('FOLD: {}, THr: {}, Dice: {}'.format(f, thr, d))

    #     all_preds.append(predicted_p)
    #     all_truths.append(truth)
    #
    # oof_pred = np.concatenate(all_preds)
    # oof_gt = np.concatenate(all_truths)

    return stats, fold2epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-i', type=str, default='')
    args = parser.parse_args()
    stats, fold2epochs = oof_eval_run(args.i)
    for k, v in fold2epochs.items():
        print('Checkpoint used for fold {} is {}'.format(k, v))
    # print(fold2epochs)
    for thr in [0.15, 0.3, 0.45, 0.6]:
        print('THr: {}, Dice: {}'.format(thr, np.mean(stats[thr])))
