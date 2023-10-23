try:
    import segmentation_models_pytorch as smp
except:
    pass

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class B3Unetpp(smp.UnetPlusPlus):
    def __init__(self, size, n_class=1, out_dim=1, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.UnetPlusPlus.__init__(self, encoder_name="efficientnet-b3",
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

    @property
    def model(self):
        return self.encoder


class B3Unet(smp.Unet):
    def __init__(self, size, n_class=1, out_dim=1, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.Unet.__init__(self, encoder_name="efficientnet-b3",
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

        # if pool == 'AdaptiveAvgPool2d':
        #     self.pooling = nn.AdaptiveAvgPool2d(1)
        # elif pool == 'gem':
        #     self.pooling = GeM()
        #
        # n_features = self.encoder.out_channels[-1]
        # self.fc = nn.Linear(n_features, out_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.classification_head = nn.Sequential(self.pooling, nn.Flatten(), self.dropout, self.fc)
        # csp = Conv2dStaticSamePadding(in_channels=4,
        #                               out_channels=self.encoder._conv_stem.out_channels,
        #                               kernel_size=self.encoder._conv_stem.kernel_size,
        #                               stride=self.encoder._conv_stem.stride, image_size=size)
        # self.decoder = nn.Identity()
        # self.encoder._conv_stem = csp

    @property
    def model(self):
        return self.encoder


class B5Unet(smp.Unet):
    def __init__(self, size, n_class=1, out_dim=19, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.Unet.__init__(self, encoder_name="efficientnet-b5",
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

        # if pool == 'AdaptiveAvgPool2d':
        #     self.pooling = nn.AdaptiveAvgPool2d(1)
        # elif pool == 'gem':
        #     self.pooling = GeM()
        #
        # n_features = self.encoder.out_channels[-1]
        # self.fc = nn.Linear(n_features, out_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.classification_head = nn.Sequential(self.pooling, nn.Flatten(), self.dropout, self.fc)
        # csp = Conv2dStaticSamePadding(in_channels=4,
        #                               out_channels=self.encoder._conv_stem.out_channels,
        #                               kernel_size=self.encoder._conv_stem.kernel_size,
        #                               stride=self.encoder._conv_stem.stride, image_size=size)
        # self.encoder._conv_stem = csp


    @property
    def model(self):
        return self.encoder
    #
    # def infer(self, x):
    #     """Sequentially pass `x` trough model`s encoder, decoder and heads"""
    #     features = self.encoder(x)
    #     return self.classification_head(features[-1])


class B7Unet(smp.Unet):
    def __init__(self, size, n_class=1, out_dim=19, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.Unet.__init__(self, encoder_name="efficientnet-b7",
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

        # if pool == 'AdaptiveAvgPool2d':
        #     self.pooling = nn.AdaptiveAvgPool2d(1)
        # elif pool == 'gem':
        #     self.pooling = GeM()
        #
        # n_features = self.encoder.out_channels[-1]
        # self.fc = nn.Linear(n_features, out_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.classification_head = nn.Sequential(self.pooling, nn.Flatten(), self.dropout, self.fc)
        #
        # csp = Conv2dStaticSamePadding(in_channels=4,
        #                               out_channels=self.encoder._conv_stem.out_channels,
        #                               kernel_size=self.encoder._conv_stem.kernel_size,
        #                               stride=self.encoder._conv_stem.stride, image_size=size)
        # self.encoder._conv_stem = csp


    @property
    def model(self):
        return self.encoder

    # def infer(self, x):
    #     """Sequentially pass `x` trough model`s encoder, decoder and heads"""
    #     features = self.encoder(x)
    #     return self.classification_head(features[-1])


class MitUnet(smp.Unet):
    def __init__(self, encoder_name, n_class=1, out_dim=19, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.Unet.__init__(self, encoder_name=encoder_name,
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

    @property
    def model(self):
        return self.encoder


class B0Unet(smp.Unet):
    def __init__(self, n_class=4, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        smp.Unet.__init__(self, encoder_name="efficientnet-b0",  # decoder_channels=(128, 64, 32, 16, 8),
                          encoder_weights="imagenet", in_channels=3, classes=n_class)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        n_features = self.encoder.out_channels[-1]

        self.bowel_head = nn.Linear(n_features, 2)
        self.extravasation_head = nn.Linear(n_features, 2)
        self.kidney_head = nn.Linear(n_features, 3)
        self.liver_head = nn.Linear(n_features, 3)
        self.spleen_head = nn.Linear(n_features, 3)
        self.any_head = nn.Linear(n_features, 2)

        self.dropout = nn.Dropout(dropout)

    #         self.classification_head = nn.Sequential(self.pooling, nn.Flatten(), self.dropout, self.fc)

    @property
    def model(self):
        return self.encoder

    def infer(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        return self.classification_head(features[-1])

    def forward(self, x, N):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        x = self.pooling(features[-1])
        x = self.dropout(nn.Flatten()(x))
        x = x.reshape(-1, N, x.shape[-1]).max(1)[0]
        return (masks, self.bowel_head(x), self.extravasation_head(x), self.kidney_head(x),
                self.liver_head(x), self.spleen_head(x), self.any_head(x))

#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels

#         return masks


class TwoHeadUnet(smp.Unet):
    def __init__(self, encoder_name='densenet121', n_class=4, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):

        if encoder_name == 'dpn92':
            weight = 'imagenet+5k'
        else:
            weight = 'imagenet'

        smp.Unet.__init__(self, encoder_name=encoder_name, decoder_channels=(128, 64, 32, 16, 8),
                          encoder_weights=weight, in_channels=3, classes=n_class)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        n_features = self.encoder.out_channels[-1]

        self.bowel_head = nn.Linear(n_features, 2)
        self.extravasation_head = nn.Linear(n_features, 2)
        self.kidney_head = nn.Linear(n_features, 3)
        self.liver_head = nn.Linear(n_features, 3)
        self.spleen_head = nn.Linear(n_features, 3)
        self.any_head = nn.Linear(n_features, 2)

        self.dropout = nn.Dropout(dropout)

    #         self.classification_head = nn.Sequential(self.pooling, nn.Flatten(), self.dropout, self.fc)

    @property
    def model(self):
        return nn.ModuleList([self.encoder, self.decoder])

    def infer(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        return self.classification_head(features[-1])

    def forward(self, x, N):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        x = self.pooling(features[-1])
        x = self.dropout(nn.Flatten()(x))
        x = x.reshape(-1, N, x.shape[-1]).max(1)[0]
        return (masks, self.bowel_head(x), self.extravasation_head(x), self.kidney_head(x),
                self.liver_head(x), self.spleen_head(x), self.any_head(x))

#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels

#         return masks