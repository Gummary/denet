# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

from typing import List

import numpy as np
import torch
import torch.nn as nn

from model.blurpool import BlurPool
from model.ops import SALayer, CALayer


class SKFF(nn.Module):

    def __init__(self,
                 input_channels: int,
                 reduction: int = 8,
                 num_branch: int = 3,
                 bias: bool = False):
        super().__init__()
        self.num_branch = num_branch

        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        d = input_channels // reduction
        self.conv_du = nn.Sequential(nn.Conv2d(input_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList()
        for i in range(self.num_branch):
            self.fcs.append(nn.Conv2d(d, input_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: List[torch.Tensor]):
        assert len(x) == self.num_branch

        batch_size = x[0].shape[0]
        num_channels = x[0].shape[1]

        feature = torch.cat(x, dim=1)
        feature = feature.view(batch_size, self.num_branch, num_channels, feature.shape[2], feature.shape[3])

        feature_sum = torch.sum(feature, dim=1)
        feature_pooled = self.average_pooling(feature_sum)
        feature_red = self.conv_du(feature_pooled)

        attention = [fc(feature_red) for fc in self.fcs]
        attention = torch.cat(attention, dim=1)
        attention = attention.view(batch_size, self.num_branch, num_channels, 1, 1)

        weights = self.softmax(attention)

        return torch.sum(feature * weights, dim=1)


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                 nn.PReLU(),
                                 BlurPool(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(BlurPool(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size // 2, stride=1),
                        act,
                        nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size // 2, stride=1)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = SALayer()

        ## Channel Attention
        self.CA = CALayer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class MRB(nn.Module):

    def __init__(self,
                 input_channels: int,
                 width: int = 2,
                 stride: int = 2,
                 num_branch: int = 3):
        super().__init__()
        self.num_branch = num_branch
        self.stride = stride
        self.width = width

        num_channels = [input_channels * stride ** i for i in range(num_branch)]
        scales = [2 ** i for i in range(1, num_branch)]
        scales.reverse()

        self.DAUs = []
        for i in range(width):
            self.DAUs.append(nn.ModuleList([DAU(channels) for channels in num_channels]))
        self.DAUs = nn.ModuleList(self.DAUs)

        self.down = nn.ModuleDict()
        for i, channels in enumerate(num_channels):
            for scale in scales[i:]:
                self.down[f"{channels}_{scale}"] = DownSample(channels, scale, stride)

        self.up = nn.ModuleDict()
        for i, channels in enumerate(reversed(num_channels)):
            for scale in scales[i:]:
                self.up[f"{channels}_{scale}"] = UpSample(channels, scale, stride)

        self.last_up = nn.ModuleDict()
        scales.reverse()
        for i in range(1, num_branch):
            self.last_up[f"{i}"] = UpSample(num_channels[i], scales[i - 1], stride)

        self.skffs = nn.ModuleList([SKFF(num_channels[i], num_branch=num_branch) for i in range(num_branch)])

        self.conv_out = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):

        TENSORS = []
        for i in range(self.num_branch):
            if i == 0:
                TENSORS.append(self.DAUs[0][i](x))
            else:
                scale = self.stride ** i
                TENSORS.append(self.DAUs[0][i](self.down[f'{x.size(1)}_{scale}'](x)))

        for w in range(1, self.width):
            fused_tensor = []
            for i in range(self.num_branch):
                tmp = []
                for j in range(self.num_branch):
                    # iterator TENSORS to get out tensor for branch i
                    scale = self.stride ** abs(i - j)
                    if j > i:
                        tmp.append(self.up[f'{TENSORS[j].size(1)}_{scale}'](TENSORS[j]))
                    elif j == i:
                        tmp.append(TENSORS[j])
                    else:
                        tmp.append(self.down[f'{TENSORS[j].size(1)}_{scale}'](TENSORS[j]))
                # use skff to fuse tensor
                fused_tensor.append(self.skffs[i](tmp))
            for i in range(self.num_branch):
                TENSORS[i] = self.DAUs[w][i](fused_tensor[i])

        for i in range(self.num_branch):
            if i == 0:
                continue
            TENSORS[i] = self.last_up[f"{i}"](TENSORS[i])

        out = x + self.conv_out(self.skffs[0](TENSORS))
        return out


class RRG(nn.Module):

    def __init__(self, input_channels: int, num_mrb: int, width: int = 2, stride: int = 2, num_branch: int = 3,
                 bias: bool = False):
        super().__init__()
        self.body = [MRB(input_channels, width, stride, num_branch) for _ in range(num_mrb)]
        self.body.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=1, bias=bias))
        self.body = nn.Sequential(*self.body)

    def forward(self, x: torch.Tensor):
        return x + self.body(x)


class Net(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.head = nn.Conv2d(opt.input_channels, opt.num_channels, kernel_size=3, padding=1, stride=1, bias=opt.bias)
        self.body = nn.Sequential(
            *[RRG(opt.num_channels, opt.num_mrb, opt.num_branch, opt.width, opt.stride, opt.bias) for _ in
              range(opt.num_rrg)])

        self.tail = nn.Sequential(*[
            # Upsampler(scale=opt.scale, num_channels=opt.num_channels),
            nn.Conv2d(opt.num_channels, opt.output_channels, kernel_size=3, padding=1, stride=1, bias=opt.bias)
        ])
        self.opt = opt

    def forward(self, x):
        x = self.head(x)
        residual = self.body(x)
        feat = residual + x
        out = self.tail(feat)

        if self.opt.with_dc:
            return feat, out

        return out
