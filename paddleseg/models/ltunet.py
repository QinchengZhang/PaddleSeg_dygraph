# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-03-31 10:27:53
LastEditors: TJUZQC
LastEditTime: 2021-04-06 16:12:34
Description: None
'''
from typing import Iterable

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models.layers import SwinTransformer
from paddleseg.models.layers.layer_libs import ConvBN, ConvBNReLU
from paddleseg.utils.einops.layers.paddle import Rearrange


@manager.MODELS.add_component
class SwinTransUNet(nn.Layer):
    def __init__(self,
                 hidden_dim: int,
                 layers,
                 heads,
                 n_channels: int = 3,
                 num_classes: int = 2,
                 head_dim=32, window_size=8,
                 downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True,
                 align_corners=False,
                 use_deconv=True):
        # Call super constructor
        super(SwinTransUNet, self).__init__()
        self.encoder = SwinTransformer(hidden_dim=hidden_dim,
                                       layers=layers,
                                       heads=heads,
                                       channels=n_channels,
                                       num_classes=num_classes,
                                       head_dim=head_dim,
                                       window_size=window_size,
                                       downscaling_factors=downscaling_factors,
                                       relative_pos_embedding=relative_pos_embedding)
        self.decoder = Decoder(align_corners=align_corners,use_deconv=use_deconv)
        self.use_deconv = use_deconv
        self.align_corners = align_corners
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                128,
                64,
                kernel_size=2,
                stride=2,
                padding=0)
            self.cls = nn.Conv2D(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding=1)
        else:
            self.cls = nn.Conv2D(
                in_channels=128,
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding=1)

    def forward(self, x):
        x,short_cuts = self.encoder(x)
        x = self.decoder(x, short_cuts)
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return [self.cls(x)]

class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[1024, 512], [512, 256], [256, 128], [128, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        # print("input size:", x.shape)
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
            # print("up sample:", x.shape)
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = out_channels
        else:
            pass

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels+out_channels, in_channels, 3),
            ConvBNReLU(in_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                short_cut.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x
