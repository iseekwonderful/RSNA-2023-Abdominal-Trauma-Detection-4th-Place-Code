import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import timm


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficinetNetFN(nn.Module):
    '''
    Version with feature linear!


    '''
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, feature_dim=512):
        super().__init__()
        # self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
        #                             pretrained=(pretrained == 'imagenet'))
        print(name)
        self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
        self.last_linear = nn.Linear(in_features=feature_dim, out_features=out_features)
        self.pool = GeM()
        self.dropout = dropout

    def forward(self, x, infer=False):
        x = self.model.features(x)
        f = self.feature_linear(nn.Flatten()(self.pool(x)))
        if infer:
            return self.pool(x)
        else:
            f = nn.ReLU()(f)
            if self.dropout:
                return self.last_linear(nn.Dropout(self.dropout)(f))
            else:
                return self.last_linear(f)


class EfficinetNet(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=14, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        # self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
        #                             pretrained=(pretrained == 'imagenet'))
        self.model = timm.create_model(name, pretrained=True)
        print(name)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()

    def forward(self, x, infer=False):
        x = self.model(x)
        # x = self.model.features(x)
        x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)


class Convnext(nn.Module):
    def __init__(self, name='convnext_base', pretrained='imagenet', out_features=14, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        in_features = self.model.head.fc.in_features
        self.model.head = torch.nn.Identity()
        self.last_linear = nn.Linear(in_features=in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, infer=False):
        x = self.model.stem(x)
        x = self.model.stages(x)
        x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)


class Swin(nn.Module):
    def __init__(self, name='swin_base_patch4_window12_384', pretrained='imagenet', out_features=14, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        in_features = self.model.head.in_features
        # self.model.avgpool = torch.nn.Identity()
        self.model.head = torch.nn.Identity()
        self.last_linear = nn.Linear(in_features=in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, infer=False):
        # x = self.model.patch_embed(x)
        # x = self.model.pos_drop(x)
        # x = self.model.layers(x)
        x = self.model(x)
        # x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)



class EfficinetNetV2(nn.Module):
    def __init__(self, name='efficientnetv2_s', pretrained='imagenet', out_features=14, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        print(name)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        # self.model.classifier = nn.Sequential(self.dropout, self.last_linear)
        # self.model.train()
        # # self.model.global_pool = self.pooling
        # for x in self.model.parameter():
        #     print(x.requires_grad)

    def forward(self, x, infer=False):
        x = self.model(x)
        x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)


class PartialEffb0(nn.Module):
    def __init__(self, level=4, dropout=0.5):
        super(PartialEffb0, self).__init__()

        print('[ AUX model ] dropout: {}'.format(dropout))
        e = timm.models.__dict__['tf_efficientnet_b0'](pretrained=True)
        self.model = e
        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]

        self.level = level
        self.b8 = nn.Sequential(
            e.conv_head,  # 384, 1536
            e.bn2,
            e.act2,
        )
        self.size2level = {
            0: 32, 1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320, 8: 1280
        }
        self.level2blocks = {
            0: self.b0, 1: self.b1, 2: self.b2, 3: self.b3,
            4: self.b4, 5: self.b5, 6: self.b6, 7: self.b7, 8: self.b8
        }
        self.logit = nn.Linear(self.size2level[level], 14)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):
        batch_size = image.shape[0]
        x = image

        #         x = self.b0(x) #; print (x.shape)  # torch.Size([2, 40, 256, 256])
        #         print(f'out - 0: {x.shape}')
        #         x = self.b1(x) #; print (x.shape)  # torch.Size([2, 24, 256, 256])
        #         print(f'out - 1: {x.shape}')
        #         x = self.b2(x) #; print (x.shape)  # torch.Size([2, 32, 128, 128])
        #         print(f'out - 2: {x.shape}')
        #         x = self.b3(x) #; print (x.shape)  # torch.Size([2, 48, 64, 64])
        #         print(f'out - 3: {x.shape}')
        #         x = self.b4(x) #; print (x.shape)  # torch.Size([2, 96, 32, 32])
        #         print(f'out - 4: {x.shape}')
        #         x = self.b5(x) #; print (x.shape)  # torch.Size([2, 136, 32, 32])
        #         print(f'out - 5: {x.shape}')
        #         x = self.b6(x) #; print (x.shape)  # torch.Size([2, 232, 16, 16])
        #         print(f'out - 6: {x.shape}')
        #         x = self.b7(x) #; print (x.shape)  # torch.Size([2, 1536, 16, 16])
        #         print(f'out - 7: {x.shape}')
        #         x = self.b8(x) #; print (x.shape)  # torch.Size([2, 1536, 16, 16])
        #         print(f'out - 8: {x.shape}')

        for i in range(0, self.level + 1):
            x = self.level2blocks[i](x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(self.dropout(x))
        return logit

