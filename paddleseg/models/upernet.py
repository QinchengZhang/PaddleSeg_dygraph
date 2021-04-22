# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-04-20 16:14:04
LastEditors: TJUZQC
LastEditTime: 2021-04-21 16:41:11
Description: None
'''
from paddleseg.models.layers.layer_libs import ConvBNReLU
import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.layers import SwinTransformer
from paddle.nn import functional as F


@manager.MODELS.add_component
class UperNet(nn.Layer):
    def __init__(self, num_classes):
        super(UperNet, self).__init__()
        self.encoder = SwinTransformer()
        self.decoder = UPerHead([96], 64)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

class UPerHead(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels, channels, num_classes, pool_scales=(1, 2, 3, 6), align_corners=False):
        super(UPerHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = layers.PPModule(
            bin_sizes=pool_scales,
            in_channels=self.in_channels[-1],
            out_channels=self.channels,
            align_corners=self.align_corners)
        self.bottleneck = ConvBNReLU(self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels, kernel_size=3,padding=1)
        # FPN Module
        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvBNReLU( #ConvBNReLU
                in_channels,
                self.channels,
                1)
            fpn_conv = ConvBNReLU(
                self.channels,
                self.channels,
                3)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvBNReLU(
            len(self.in_channels) * self.channels,
            self.channels,
            3)
        self.cls = nn.Conv2D(channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        x1 = self.psp_modules(x)
        psp_outs.append(x1)
        psp_outs = paddle.concat(psp_outs, axis=1)
        print(psp_outs.shape)
        output = self.bottleneck(psp_outs)

        return output
    

    def forward(self, inputs):
        """Forward function."""


        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i],prev_shape,mode='bilinear',align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i],size=fpn_outs[0].shape[2:],mode='bilinear',align_corners=self.align_corners)
        fpn_outs = paddle.concat(fpn_outs, axis=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls(output)
        return output
