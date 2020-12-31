"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import math

import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(
            self,
            rgb_range, sign=-1,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, num_channels, res_scale=1.0):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, num_channels, scale):
        m = list()
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_channels, 4 * num_channels, 3, 1, 1)]
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m += [nn.Conv2d(num_channels, 9 * num_channels, 3, 1, 1)]
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super().__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale ** 2), h // self.scale, w // self.scale)
        return x


class CALayer(nn.Module):
    def __init__(self, num_channels, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SALayer(nn.Module):
    def __init__(self, kernel_size=5):
        super(SALayer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale
