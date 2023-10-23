from utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score
from configs import Config
import torch
from utils import rand_bbox
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
import pickle as pk
from path import Path
from utils.metric import score, train_loss

import os
try:
    from apex import amp
except:
    pass
import torch.distributed as dist
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from utils.metric import macro_multilabel_auc, pfbeta


def to_hex(image_id) -> str:
    return '{0:0{1}x}'.format(image_id, 12)


def gather_list_and_concat(list_of_nums):
    tensor = torch.Tensor(list_of_nums).cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def gather_tensor_and_concat(tensor):
    tensor = tensor.cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def read_data(data):
    return tuple(d.cuda() for d in data[:-3]), data[-3].cuda(), data[-2].cuda(), data[-1]


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, gpu, tune=None):
    # if gpu == 0:
    #     print('!!!!!!!!!!!!!!!!!', save_path)
    device = torch.device(f'cuda:{gpu}')
    if gpu == 0:
        print('[ √ ] Basic training')
    # focal = torch.nn.BCEWithLogitsLoss()
    two_class_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 6.]).to(device))
    three_class_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4.]).to(device))
    if type(cfg.loss.ch_weight) == int:
        focal = torch.nn.BCEWithLogitsLoss(pos_weight=(torch.ones(4*int(cfg.transform.size)*int(cfg.transform.size))*50).to(device))
    else:
        print('Channel weight: ', cfg.loss.ch_weight)
        focal = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.cat([torch.ones(int(cfg.transform.size)*int(cfg.transform.size))*cfg.loss.ch_weight[0],
                                  torch.ones(int(cfg.transform.size)*int(cfg.transform.size))*cfg.loss.ch_weight[1],
                                  torch.ones(int(cfg.transform.size)*int(cfg.transform.size))*cfg.loss.ch_weight[2],
                                  torch.ones(int(cfg.transform.size)*int(cfg.transform.size))*cfg.loss.ch_weight[3]]).to(device)
        )

    # focal = torch.nn.BCEWithLogitsLoss(pos_weight=(torch.ones(4*384*384)*50).cuda())
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            if epoch == 0 and cfg.train.freeze_start_epoch:
                print('[ W ] Freeze backbone layer')
                # only fit arcface-efficient model
                for x in model.module.model.parameters():
                    x.requires_grad = False
            if epoch == 1 and cfg.train.freeze_start_epoch:
                print('[ W ] Unfreeze backbone layer')
                for x in model.module.model.parameters():
                    x.requires_grad = True
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if gpu == 0:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses, cls_losses, seg_losses = [], [], []
            pos_cases, neg_casses = [], []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            for i, data in enumerate(tq):
                # ignore last for valid
                data = data[:-1]
                # print(label_image)
                # print(lbl_study)
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                inputs, labels, mask = data[0], [e.to(device) for e in data[1:-1]], data[-1]
                bs, N, d, w, h = inputs.shape
                inputs = inputs.reshape(bs * N, d, w, h).to(device)
                mask = mask.reshape(bs * N, 4, w, h).to(device)

                r = np.random.rand(1)
                if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    input, target_a, target_b, lam_a, lam_b = cutmix(img, lbl, cfg.train.beta)
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            if 'dist' in cfg.model.name:
                                cls = model(input, lbl)
                            else:
                                cls = model(input)
                        cls = cls.float()
                    else:
                        cls = model(input)
                    # cls loss
                    # print(loss_func(cls, target_a).mean(1).shape)
                    # print(torch.tensor(
                    #     lam_a).cuda().float().shape)
                    if cls.shape[-1] != 256:
                        cls = torch.nn.functional.interpolate(cls, size=(256, 256))
                    cls_loss = (loss_func(cls, target_a).mean())
                        #         * torch.tensor(
                        # lam_a).cuda().float() +
                        #     loss_func(cls, target_b).mean() * torch.tensor(
                        #         lam_b).cuda().float())
                    if not len(cls_loss.shape) == 0:
                        cls_loss = cls_loss.mean()
                    # bce_loss = torch.nan_to_num(bce_loss)
                    loss = cls_loss
                else:
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs, N)
                        loss_cls = train_loss([o.float() for o in outputs[1:]], [l.long() for l in labels], two_class_ce, three_class_ce, cfg.loss.sub_weight)
                        # print(mask.shape, outputs[0].shape)
                        seg_loss = focal(outputs[0].reshape(mask.shape[0], -1), mask.float().reshape(mask.shape[0], -1))
                        loss = 1 * loss_cls + cfg.loss.seg_weight * seg_loss
                        cls_losses.append(loss_cls.item())
                        seg_losses.append(seg_loss.item())
                    else:
                        cls = model(img)
                        if cls.shape[-1] != 256:
                            cls = torch.nn.functional.interpolate(cls, size=(256, 256))
                        cls_loss = loss_func(cls.float(), lbl)
                        # mse_losses.append(mse_loss.item())
                        if not len(cls_loss.shape) == 0:
                            cls_loss = cls_loss.mean()
                        # bce_loss = torch.nan_to_num(bce_loss)
                        loss = cls_loss
                losses.append(loss.item())
                # cutmix ended
                # output = model(ipt)
                # loss = loss_func(output, lbl)
                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # predicted.append(output.detach().sigmoid().cpu().numpy())
                # truth.append(lbl.detach().cpu().numpy())
                if i % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                            scheduler.step(epoch + i / len(train_dl))
                        else:
                            scheduler.step()
                if gpu == 0:
                    tq.set_postfix(loss=np.array(losses).mean(), cls_loss=np.mean(cls_losses),
                                   seg_loss=np.mean(seg_losses),
                                   lr=optimizer.param_groups[0]['lr'])
            if len(valid_dl) > 0:
                validate_loss, accuracy, auc, bce_loss, mse_loss, ap, f103, f105, pf1 = basic_validate(model,  valid_dl, loss_func, cfg, gpu, tune)
            else:
                print('[ ! ] Skip validation!')
                validate_loss, accuracy, auc, bce_loss, mse_loss, ap, f103, f105, pf1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            if tune:
                tune.report(valid_loss=validate_loss, valid_auc=auc, train_loss=np.mean(losses),
                            seg_loss=bce_loss, cls_loss=mse_loss)
            if gpu == 0:
                print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}, ' +
                       'dice: {:.4f}, bowel: {:.4f}, kidney: {:.4f}, extravasation: {:.4f}, liver: {:.4f} ' +
                       'spleen: {:.4f}, any: {:.4f}, score: {:.4f}'
                       ).format(
                    epoch, np.array(losses).mean(), validate_loss, accuracy, auc, float(mse_loss),
                    float(bce_loss), ap, f103, f105, pf1))
            if writer is not None:
                writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
                writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
                writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
                writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
                writer.add_scalar('valid_f{}/pF1@0.3'.format(cfg.experiment.run_fold), ap, epoch)
                writer.add_scalar('valid_f{}/pF1@0.5'.format(cfg.experiment.run_fold), f103, epoch)
                writer.add_scalar('valid_f{}/pF1@0.65'.format(cfg.experiment.run_fold), f105, epoch)
                writer.add_scalar('valid_f{}/pF1@0.8'.format(cfg.experiment.run_fold), pf1, epoch)
            if gpu == 0:
                with open(save_path / 'train.log', 'a') as fp:
                    fp.write(
                        '{}\t{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                            cfg.experiment.run_fold, epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(),
                            validate_loss, accuracy, auc, float(mse_loss), float(bce_loss), ap, f103, f105, pf1))
            if gpu == 0:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


# def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, gpu, tune=None):
#     if cfg.basic.amp == 'Native':
#         scaler = torch.cuda.amp.GradScaler()
#     if gpu == 0:
#         print('[ √ ] Basic training')
#     try:
#         optimizer.zero_grad()
#         for epoch in range(cfg.train.num_epochs):
#             # set epoch
#             # sampler.set_epoch(epoch)
#             train_dl.sampler.set_epoch(epoch)
#
#             if epoch == 0 and cfg.train.freeze_start_epoch:
#                 print('[ W ] Freeze backbone layer')
#                 # only fit arcface-efficient model
#                 for x in model.module.model.parameters():
#                     x.requires_grad = False
#             if epoch == 1 and cfg.train.freeze_start_epoch:
#                 print('[ W ] Unfreeze backbone layer')
#                 for x in model.module.model.parameters():
#                     x.requires_grad = True
#             # first we update batch sampler if exist
#             # if cfg.experiment.batch_sampler:
#             #     train_dl.batch_sampler.update_miu(
#             #         cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
#             #     )
#             #     print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
#             if scheduler and cfg.scheduler.name in ['StepLR']:
#                 scheduler.step(epoch)
#             model.train()
#             if gpu == 0:
#                 tq = tqdm.tqdm(train_dl)
#             else:
#                 tq = train_dl
#             basic_lr = optimizer.param_groups[0]['lr']
#             cls_losses, slice_losses = [], []
#             losses = []
#             pos_cases, neg_casses = [], []
#             # native amp
#             for i, (img, lbl) in enumerate(tq):
#                 if cfg.scheduler.warm_up and epoch == 0:
#                     # warm up
#                     length = len(train_dl)
#                     initial_lr = basic_lr / length
#                     optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
#
#                 img, lbl = img.cuda(), lbl.cuda()
#                 pos_cases.append((lbl == 1).sum().cpu().item())
#                 neg_casses.append((lbl == 0).sum().cpu().item())
#
#                 r = np.random.rand(1)
#                 if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
#                     input, target_a, target_b, lam_a, lam_b = cutmix(img, lbl, cfg.train.beta)
#                     if cfg.basic.amp == 'Native':
#                         with torch.cuda.amp.autocast():
#                             if 'dist' in cfg.model.name:
#                                 cls = model(input, lbl)
#                             else:
#                                 cls = model(input)
#                         cls = cls.float()
#                     else:
#                         cls = model(input)
#                     # cls loss
#                     # print(loss_func(cls, target_a).mean(1).shape)
#                     # print(torch.tensor(
#                     #     lam_a).cuda().float().shape)
#                     cls_loss = (loss_func(cls, target_a).mean() * torch.tensor(
#                         lam_a).cuda().float() +
#                             loss_func(cls, target_b).mean() * torch.tensor(
#                                 lam_b).cuda().float())
#                     if not len(cls_loss.shape) == 0:
#                         cls_loss = cls_loss.mean()
#                     # bce_loss = torch.nan_to_num(bce_loss)
#                     loss = cls_loss
#                 else:
#                     if cfg.basic.amp == 'Native':
#                         with torch.cuda.amp.autocast():
#                             if 'dist' in cfg.model.name or 'arc' in cfg.model.name:
#                                 cls = model(img, lbl)
#                             else:
#                                 cls = model(img)
#                         cls_loss = loss_func(cls.float(), lbl)
#                         # mse_losses.append(mse_loss.item())
#                         if not len(cls_loss.shape) == 0:
#                             cls_loss = cls_loss.mean()
#                         # bce_loss = torch.nan_to_num(bce_loss)
#                         loss = cls_loss
#                     else:
#                         cls = model(img)
#                         cls_loss = loss_func(cls.float(), lbl)
#                         # mse_losses.append(mse_loss.item())
#                         if not len(cls_loss.shape) == 0:
#                             cls_loss = cls_loss.mean()
#                         # bce_loss = torch.nan_to_num(bce_loss)
#                         loss = cls_loss
#
#                 losses.append(loss.item())
#                 # cutmix ended
#                 # output = model(ipt)
#                 # loss = loss_func(output, lbl)
#                 if cfg.basic.amp == 'Native':
#                     scaler.scale(loss).backward()
#                 elif not cfg.basic.amp == 'None':
#                     with amp.scale_loss(loss, optimizer) as scaled_loss:
#                         scaled_loss.backward()
#                 else:
#                     loss.backward()
#
#                 if i % cfg.optimizer.step == 0:
#                     if cfg.basic.amp == 'Native':
#                         if cfg.train.clip:
#                             scaler.unscale_(optimizer)
#                             # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
#                             torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
#                         scaler.step(optimizer)
#                         scaler.update()
#                         optimizer.zero_grad()
#                     else:
#                         if cfg.train.clip:
#                             torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
#                         optimizer.step()
#                         optimizer.zero_grad()
#                 if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
#                     if epoch == 0 and cfg.scheduler.warm_up:
#                         pass
#                     else:
#                         if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
#                             scheduler.step(epoch + i / len(train_dl))
#                         else:
#                             scheduler.step()
#
#                 if gpu == 0:
#                     tq.set_postfix(loss=np.array(losses).mean(), pos=np.mean(pos_cases), neg=np.mean(neg_casses),
#                                    lr=optimizer.param_groups[0]['lr'])
#             if len(valid_dl) > 0:
#                 validate_loss, accuracy, auc, bce_loss, mse_loss, ap, f103, f105, pf1 = basic_validate(model, valid_dl, loss_func, cfg, tune=tune)
#             else:
#                 print('[ ! ] Skip validation!')
#                 validate_loss, accuracy, auc, bce_loss, mse_loss, ap, f103, f105, pf1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
#             if tune:
#                 tune.report(valid_loss=validate_loss, valid_auc=auc, train_loss=np.mean(losses),
#                             seg_loss=bce_loss, cls_loss=mse_loss)
#             if gpu == 0:
#                 print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}, ' +
#                        'accuracy: {:.4f}, auc: {:.4f}, mse_loss: {:.4f}, pF1: {:.4f}, pF1@0.3: {:.4f} ' +
#                        'pF1@0.5: {:.4f}, pF1@0.65: {:.4f}, pF1@0.8: {:.4f}'
#                        ).format(
#                     epoch, np.array(losses).mean(), validate_loss, accuracy, auc, float(mse_loss),
#                     float(bce_loss), ap, f103, f105, pf1))
#             if writer:
#                 writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
#                 writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'],
#                                   epoch)
#                 writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
#                 writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
#                 writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
#                 writer.add_scalar('valid_f{}/pF1@0.3'.format(cfg.experiment.run_fold), ap, epoch)
#                 writer.add_scalar('valid_f{}/pF1@0.5'.format(cfg.experiment.run_fold), f103, epoch)
#                 writer.add_scalar('valid_f{}/pF1@0.65'.format(cfg.experiment.run_fold), f105, epoch)
#                 writer.add_scalar('valid_f{}/pF1@0.8'.format(cfg.experiment.run_fold), pf1, epoch)
#             # neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses))
#             # if not cfg.basic.debug:
#             #     neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses))
#             #     neptune.log_metric('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'])
#             #     neptune.log_metric('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss)
#             #     neptune.log_metric('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy)
#             #     neptune.log_metric('valid_f{}/auc'.format(cfg.experiment.run_fold), auc)
#             if gpu == 0:
#                 with open(save_path / 'train.log', 'a') as fp:
#                     fp.write(
#                         '{}\t{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
#                             cfg.experiment.run_fold,
#                             epoch,
#                             optimizer.param_groups[0][
#                                 'lr'],
#                             np.array(losses).mean(),
#                             validate_loss, bce_loss,
#                             mse_loss, accuracy, auc, ap, f103, f105, pf1))
#
#                 torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
#                     cfg.experiment.run_fold, epoch))
#             if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
#                 scheduler.step(validate_loss)
#     except KeyboardInterrupt:
#         print('[ X ] Ctrl + c, QUIT')
#         torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def basic_validate(mdl,  dl, loss_func,  cfg, gpu, tune=None):
    device = torch.device(f'cuda:{gpu}')
    two_class_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 6.]).to(device))
    three_class_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4.]).to(device))

    focal = torch.nn.BCEWithLogitsLoss(pos_weight=(torch.ones(4 * int(cfg.transform.size) * int(cfg.transform.size)) * 50).to(device))

    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, predicted_p, truth = [], [], [], []
        original_truth = []
        cls_losses, bce_losses = [], []
        dices = []
        gts = []
        if gpu == 0:
            tq = tqdm.tqdm(dl)
        else:
            tq = dl
        for i, data in enumerate(tq):
            data, gt = data[:-1], data[-1]
            gts.append(gt.cpu().numpy())
            inputs, labels, mask = data[0], [e.to(device) for e in data[1:-1]], data[-1]
            bs, N, d, w, h = inputs.shape
            inputs = inputs.reshape(bs * N, d, w, h).to(device)
            mask = mask.reshape(bs * N, 4, w, h).to(device)
            # img, lbl = img.cuda(), label_image.cuda()
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    o = mdl(inputs, N)
                    outputs = o[1:]
                    out_mask = torch.sigmoid(o[0])

                dices.append(
                    ((mask.cpu() * (out_mask.cpu() > 0.3)).sum() / (
                                (mask.cpu() + (out_mask.cpu() > 0.3)) > 0.5).sum()).item())
                predicted.append(torch.cat([torch.softmax(e.float(), 1) for e in outputs], 1).cpu().numpy())

                loss_cls = train_loss([o.float() for o in o[1:]], [l.long() for l in labels], two_class_ce, three_class_ce, cfg.loss.sub_weight)
                seg_loss = focal(o[0].reshape(mask.shape[0], -1), mask.float().reshape(mask.shape[0], -1))
                # loss = 1 * loss_cls + cfg.loss.seg_weight * seg_loss

                loss = loss_cls + cfg.loss.seg_weight * seg_loss

            losses.append(loss.item())
            # predicted.append(torch.softmax(cls.float().cpu(), 1).numpy())
            # truth.append(lbl.cpu().numpy())
            results.append({
                'step': i,
                'loss': loss.item(),
            })

        pred = np.concatenate(predicted)
        gt = np.concatenate(gts)
        # print(pred.shape, gt.shape)
        # calculate losses
        val_loss = np.array(losses).mean()
        val_losses = gather_list_and_concat([val_loss])
        collected_loss = val_losses.cpu().numpy().mean()

        # calculate dice
        dices = np.array(dices).mean()
        dices = gather_list_and_concat([dices])
        dice = dices.cpu().numpy().mean()
        # if gpu == 0:
        #     print(f'DICE: {dice}')
        # calculate score here
        cgt = gather_list_and_concat(gt).cpu().numpy()
        cprd = gather_list_and_concat(pred).cpu().numpy()

        # print(cprd.shape, cgt.shape)

        pred_df = pd.DataFrame(cprd[:, :13], columns=dl.dataset.df.columns[4:-3])
        gt = pd.DataFrame(cgt, columns=dl.dataset.df.columns[4:-2])
        pred_df['any_injury'] = cprd[:, -1]

        gt['file'] = 1
        pred_df['file'] = 1


        # gt = dl.dataset.df.copy()
        # print(gt.shape, pred.shape)
        # # gt_columns = ['series', 'bowel_healthy', 'bowel_injury', 'extravasation_healthy',
        # #               'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy',
        # #               'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high', 'any_injury']
        #
        # # print(dl.dataset.df.columns)
        # pred_df = pd.DataFrame(pred[:, :13], columns=dl.dataset.df.columns[4:-3])
        # pred_df['any_injury'] = pred[:, -1]

        bowel_weight = gt['bowel_injury'].values + 1
        extravasation_weight = gt.extravasation_injury.values * 5 + 1
        kidney_weight = (gt.kidney_healthy + gt.kidney_low * 2 + gt.kidney_high * 4).values
        liver_weight = (gt.liver_healthy + gt.liver_low * 2 + gt.liver_high * 4).values
        spleen_weight = (gt.spleen_healthy + gt.spleen_low * 2 + gt.spleen_high * 4).values
        any_injury_weight = (gt.any_injury * 5 + 1).values

        gt['bowel_weight'] = bowel_weight
        gt['extravasation_weight'] = extravasation_weight
        gt['kidney_weight'] = kidney_weight
        gt['liver_weight'] = liver_weight
        gt['spleen_weight'] = spleen_weight
        gt['any_injury_weight'] = any_injury_weight

        pred_df['file'] = gt['file'].values
        scores, sub = score(gt.reset_index(), pred_df.drop('any_injury', axis=1), 'file', False)
        if gpu == 0:
            print('EVAL, loss_score: {}, dice: {}'.format(scores, dice))

        return collected_loss, dice, sub['bowel'], sub['extravasation'], sub['kidney'], sub['liver'], sub[
            'spleen'], sub['any'], scores



        # predicted = gather_list_and_concat(predicted)
        # predicted = predicted.cpu().numpy()
        # truth = gather_list_and_concat(truth)
        # truth = truth.cpu().numpy()
        #
        # accuracy = (predicted.argmax(1) == truth).mean()
        # # auc = macro_multilabel_auc(truth, predicted, gpu=-1)
        # auc = roc_auc_score(truth, predicted[:, 1])
        # # ap = average_precision_score(truth, predicted, average=None)
        # ap = 0
        # # print([round(x, 3) for x in ap])
        # # ap = np.mean(ap)
        # # print(truth.shape, predicted[:, 1].shape)
        # ap = pfbeta(truth, predicted[:, 1] > 0.3, 1)
        # f103 = pfbeta(truth, predicted[:, 1] > 0.5, 1)
        # f105 = pfbeta(truth, predicted[:, 1] > 0.65, 1)
        # pf1 = pfbeta(truth, predicted[:, 1] > 0.8, 1)
        #
        # pf1_raw = pfbeta(truth, predicted[:, 1], 1)
        #
        # return val_loss, accuracy, auc, pf1_raw, cls_loss, ap, f103, f105, pf1
        # # return val_loss, 0, 0, 0, 0