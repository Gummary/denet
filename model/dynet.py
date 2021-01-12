# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import ops
from model.adaptive_conv import AdaptiveConv2d


def reshape():
    a = torch.range(1, 16)
    a = a.view(2, 2, 2, 2)
    b = torch.cat([a[i, :, :, :] for i in range(2)], dim=1)
    b = torch.cat([b[i, :, :] for i in range(2)], dim=1)
    assert b.shape == (4, 4)


class PixelConv(nn.Module):
    # Generate pixel kernel  (3*k*k)xHxW
    def __init__(self, in_feats, out_feats=3, rate=4, ksize=3, scale=1):
        super(PixelConv, self).__init__()
        self.in_feats = in_feats
        self.padding = (ksize - 1) // 2
        self.ksize = ksize
        self.scale = scale
        self.zero_padding = nn.ZeroPad2d(self.padding)
        mid_feats = in_feats * rate ** 2
        self.kernel_conv = nn.Sequential(*[
            nn.Conv2d(in_feats, mid_feats, kernel_size=3, padding=1),
            nn.Conv2d(mid_feats, mid_feats, kernel_size=3, padding=1),
            nn.Conv2d(mid_feats, 3 * ksize ** 2 * scale ** 2, kernel_size=3, padding=1)
        ])

    def forward(self, x_feature, x):
        kernel_set = self.kernel_conv(x_feature)

        dtype = kernel_set.data.type()
        N = self.ksize ** 2  # patch size
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)

        p = self._get_index(kernel_set, dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()
        x_pixel_set = self._get_x_q(x, p, N)
        b, c, h, w = kernel_set.size()

        if self.scale > 1:
            kernel_set_reshape = kernel_set.reshape(-1, self.scale, self.scale, self.ksize ** 2, 3, h, w).permute(0, 4,
                                                                                                                  5, 6,
                                                                                                                  1, 2,
                                                                                                                  3)
            x_pixel_set = torch.unsqueeze(x_pixel_set, dim=-2).repeat(1, 1, 1, 1, self.scale ** 2, 1).reshape(b, 3, h,
                                                                                                              w,
                                                                                                              self.scale,
                                                                                                              self.scale,
                                                                                                              self.ksize ** 2)
        else:
            kernel_set_reshape = kernel_set.reshape(-1, self.ksize ** 2, 3, h, w).permute(0, 2, 3, 4, 1)

        out = x_pixel_set * kernel_set_reshape
        out = out.sum(dim=-1, keepdim=True).squeeze(dim=-1)
        if self.scale > 1:
            out = torch.cat([out[:, :, i, :, :, :] for i in range(h)], dim=3)
            out = torch.cat([out[:, :, i, :, :] for i in range(w)], dim=3)

        return out

    def _get_index(self, kernel_set, dtype):
        '''
        get absolute index of each pixel in image
        '''
        N, b, h, w = self.ksize ** 2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
        # get absolute index of center index for each weight in conv kernel
        p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding),
                                   indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                                   range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
        p = p_0 + p_n
        p = p.repeat(b, 1, 1, 1)
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*padded_w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # index_x*w + index_y
        index = q[..., :N] * padded_w + q[..., N:]

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset


class _DynamicBlock(nn.Module):

    def __init__(self, in_channels, num_channels, scale=1):
        super().__init__()
        self.image_conv = nn.Sequential(*[
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(16, num_channels, kernel_size=3, padding=1, stride=1),
        ])
        edge_conv = [
            nn.Conv2d(num_channels * 2, 16, kernel_size=3, padding=1, stride=1)
        ]
        if scale > 1:
            edge_conv.append(ops.Upsampler(16, scale=scale))
        edge_conv.append(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))
        self.edge_conv = nn.Sequential(*edge_conv)

        self.dynamic_conv = PixelConv(num_channels * 2, scale=scale)

    def forward(self, feat, img):
        img_feat = self.image_conv(img)
        feat = torch.cat((feat, img_feat), dim=1)

        edge = self.edge_conv(feat)
        dynamic_feat = self.dynamic_conv(feat, img)

        return edge + dynamic_feat


class DynamicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1, eps=0.1):
        super().__init__()

        self.eps = eps

        self.conv_mid = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

        self.param_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, in_channels, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1))

        # Parameter factorization
        self.conv1x1_U = nn.Conv2d(in_channels, in_channels // reduction, 1, 1)
        self.conv1x1_V = nn.Conv2d(in_channels // reduction, in_channels, 1, 1)

        self.conv1x1_out = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, feat, x):
        theta = self.param_adapter(x)

        mid_feat = self.conv_mid(feat)
        dynamic_feat = self.conv1x1_U(mid_feat)
        dynamic_conv = AdaptiveConv2d(dynamic_feat.size(0) * dynamic_feat.size(1),
                                      dynamic_feat.size(0) * dynamic_feat.size(1),
                                      5, padding=theta.size(2) // 2,
                                      groups=dynamic_feat.size(0) * dynamic_feat.size(1), bias=False)
        dynamic_feat = dynamic_conv(input=dynamic_feat, dynamic_weight=theta)
        feat_delta = self.conv1x1_V(dynamic_feat)

        feat_delta = self.conv1x1_out(feat_delta)

        return x * (1 + self.eps * torch.tanh(feat_delta))

class DynamicBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1, eps=0.1):
        super().__init__()

        self.eps = eps

        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, in_channels, 3, stride=1, padding=1),
        )

        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        )

        self.param_adapter = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1)
            nn.Conv2d(in_channels, in_channels, 3, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1))

        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, feat, x):

        oneshot_feat = self.img_conv(x)
        feat = torch.cat((feat, oneshot_feat), dim=1)

        theta = self.param_adapter(feat)

        dynamic_feat = self.feat_conv(feat)
        dynamic_conv = AdaptiveConv2d(dynamic_feat.size(0) * dynamic_feat.size(1),
                                      dynamic_feat.size(0) * dynamic_feat.size(1),
                                      5, padding=theta.size(2) // 2,
                                      groups=dynamic_feat.size(0) * dynamic_feat.size(1), bias=False)
        feat_delta = dynamic_conv(input=dynamic_feat, dynamic_weight=theta)

        feat_delta = self.out_conv(feat_delta)

        return x * (1 + self.eps * torch.tanh(feat_delta))




class Net(nn.Module):

    def __init__(self, opt):
        super().__init__()

        backbone = opt.model.split("_")[0].lower()
        self.backbone = importlib.import_module(f"model.{backbone}").Net(opt)
        self.dynamicBlocks = nn.ModuleList([DynamicBlock2(opt.num_channels, 3) for _ in range(opt.num_dc)])
        self.feat_upsampler = ops.Upsampler(opt.num_channels, scale=opt.scale)
        self.opt = opt

    def forward(self, x):
        feat, out = self.backbone(x)
        outputs = [out]

        if feat.size(2) != x.size(2):
            feat = self.feat_upsampler(feat)

        for dc in self.dynamicBlocks:
            out = dc(feat, outputs[-1])
            outputs.append(out)

        return outputs
