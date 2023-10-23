from models.resnet import *
from models.xresnet import *
from models.public import MultiTailPub
from models.margins import CosineSe50, EfficinetNetArcFace, EfficinetNetCosine, ConvNextCosine, ConvNextCosine2
from models.distance import ArcModel, ArcModelTail
from models.tile_model import TileModel, ConcatModel, SLModel
from models.resnetd import RANZCRResNet200D
from models.efficient import EfficinetNet, EfficinetNetV2, PartialEffb0, Convnext, Swin
from configs import Config
# from models.unet import R200DUnet, R200DUnetS, R50DUnetS
# from models.unet_att import convert_act_cls, R200DUnetS_ATT
from models.eff_unet import B0Unet, B5Unet, B7Unet, MitUnet, B3Unetpp, TwoHeadUnet


def get_model(cfg: Config, pretrained='imagenet'):
    if 'unete' in cfg.model.name:
        backbone = '_'.join(cfg.model.name.split('_')[:-1])
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 4
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return TwoHeadUnet(encoder_name=backbone, n_class=n_class, pool=pool)
    if 'swin' in cfg.model.name:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        return Swin(cfg.model.name, out_features=cfg.model.out_feature,
                        dropout=cfg.model.param['dropout'], pool=pool)
    if 'convnext' in cfg.model.name:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        return Convnext(cfg.model.name, out_features=cfg.model.out_feature,
                        dropout=cfg.model.param['dropout'], pool=pool)
    if cfg.model.name in ['cs_dist']:
        return ConvNextCosine(name='convnext_small')
    if cfg.model.name in ['b0_arc']:
        return EfficinetNetArcFace()
    if cfg.model.name in ['cm_dist']:
        return ConvNextCosine()
    if cfg.model.name in ['cm2_dist']:
        return ConvNextCosine2()
    if cfg.model.name in ['b0_dist']:
        return EfficinetNetCosine()
    if cfg.model.name in ['b5_dist']:
        return EfficinetNetCosine(name='tf_efficientnet_b5')
    if cfg.model.name in ['partial_b0']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n = cfg.model.param.get('n', '8')
        print('Partial Effnet!')
        return PartialEffb0(n)
    if cfg.model.name in ['unet50ds']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 19
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return R50DUnetS(pool=pool, n_class=n_class)
    if 'mit' in cfg.model.name and 'unet' in cfg.model.name:
        encoder_name = cfg.model.name.replace('_unet', '')
        return MitUnet(encoder_name=encoder_name)
    if cfg.model.name in ['eb7_unet']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 1
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return B7Unet(size=cfg.transform.size, pool=pool, n_class=n_class)
    if cfg.model.name in ['eb3_upp']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 1
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return B3Unetpp(size=cfg.transform.size, pool=pool, n_class=n_class)
    if cfg.model.name in ['eb5_unet']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 1
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return B5Unet(size=cfg.transform.size, pool=pool, n_class=n_class)
    if cfg.model.name in ['eb0_unet']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = 4
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return B0Unet(pool=pool, n_class=n_class)
    # if cfg.model.name in ['unet200ds_att']:
    #     pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
    #     n_class = {1: 4, 2: 6, 3: 7}[cfg.experiment.mask_version]
    #     print(f'[ ! ] Init a R200DUnetS_ATT, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
    #     mdl = R200DUnetS_ATT(pool=pool, n_class=n_class)
    #     return convert_act_cls(mdl, nn.ReLU, nn.SiLU())
    if cfg.model.name in ['unet200ds']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        n_class = {1: 4, 2: 6, 3: 7}[cfg.experiment.mask_version]
        print(f'[ ! ] Init a R200DUnetS, mdl_name: {cfg.model.name}, pool: {pool}, n_class: {n_class}')
        return R200DUnetS(pool=pool, n_class=n_class)
    if cfg.model.name in ['unet200d']:
        return R200DUnet()
    if cfg.model.name in ['resnet200d', 'resnet152d', 'resnet101d', 'resnet50d', 'resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a RANZCRResNet200D, mdl_name: {cfg.model.name}, pool: {pool}')
        return RANZCRResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['cos_tf_efficientnet_b5', 'cos_tf_efficientnet_b0', 'cos_tf_efficientnet_b7']:
        return EfficinetNetCosine(name=cfg.model.name[4:], pretrained=pretrained,
                                   m=cfg.model.param.get('dropout', 0.1),
                                   out_features=cfg.model.out_feature, dropout=cfg.model.param['dropout'])
    if cfg.model.name in ['arc_tf_efficientnet_b5', 'arc_tf_efficientnet_b7']:
        return EfficinetNetArcFace(name=cfg.model.name[4:], pretrained=pretrained,
                                   m=cfg.model.param.get('dropout', 0.1),
                                   out_features=cfg.model.out_feature, dropout=cfg.model.param['dropout'])
    # eff
    elif cfg.model.name in ['tf_efficientnet_b5', 'efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b0', 'tf_efficientnet_b7',
                            'tf_efficientnet_l2_ns', 'tf_efficientnet_b6_ns']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        return EfficinetNet(name=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature,
                            dropout=cfg.model.param['dropout'], pool=pool)
    # effv2
    elif cfg.model.name in ['efficientnetv2_l', 'efficientnetv2_m', 'efficientnetv2_rw_s', 'efficientnetv2_s',
                            'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2',
                            'tf_efficientnetv2_b3', 'tf_efficientnetv2_l', 'tf_efficientnetv2_l_in21ft1k',
                            'tf_efficientnetv2_l_in21k', 'tf_efficientnetv2_m', 'tf_efficientnetv2_m_in21ft1k',
                            'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_s', 'tf_efficientnetv2_s_in21ft1k',
                            'tf_efficientnetv2_s_in21k']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        return EfficinetNetV2(name=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature,
                            dropout=cfg.model.param['dropout'], pool=pool)
    elif cfg.model.name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return CustomResnetModel(backbone=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature)
    elif cfg.model.name in ['densenet161', 'densenet121', 'densenet201', 'densenet169']:
        return DenseNet(backbone=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature)
    elif cfg.model.name in ['XResNet', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152',
           'xresnet18_deep', 'xresnet34_deep', 'xresnet50_deep']:
        return globals()[cfg.model.name](c_out=11)
    elif cfg.model.name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']:
        return CustomSenet(name=cfg.model.name, pretrained=pretrained)
    elif cfg.model.name in ['arc_resnet18', 'arc_resnet34', 'arc_resnet50', 'arc_se_resnext50_32x4d',
                     'arc_se_resnext101_32x4d', 'arc_senet154']:
        return ArcModel(pretrained=pretrained, model=cfg.model.name)
    elif cfg.model.name in ['art_resnet18', 'art_resnet34', 'art_resnet50', 'art_se_resnext50_32x4d',
                     'art_se_resnext101_32x4d', 'art_senet154']:
        return ArcModelTail(pretrained=pretrained, model=cfg.model.name)
    elif cfg.model.name in ['CosineSe50']:
        return CosineSe50()
