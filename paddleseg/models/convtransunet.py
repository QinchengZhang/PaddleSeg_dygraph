# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-04-22 15:55:19
LastEditors: TJUZQC
LastEditTime: 2021-04-25 19:01:35
Description: None
'''
from re import S

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional.common import upsample
from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils.einops.layers.paddle import Rearrange
from paddleseg.models.layers.ConvTransformer import Transformer


@manager.MODELS.add_component
class ConvTransUNet(nn.Layer):
    """
    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 image_size, in_channels, num_classes,
                 heads=[1, 3, 6], depth=[1, 2, 10], dropout=0., emb_dropout=0., scale_axis=4,
                 align_corners=False,
                 resample_mode='bilinear',
                 pretrained=None):
        super().__init__()

        self.encode = Encoder(image_size, in_channels,
                              heads=heads, depth=depth, dropout=dropout, emb_dropout=emb_dropout, scale_axis=scale_axis)
        self.decode = Decoder(align_corners, mode=resample_mode)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    def __init__(self, image_size, in_channels, heads=[1, 3, 6], depth=[1, 2, 4], dropout=0., emb_dropout=0., scale_axis=2):
        super().__init__()

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(3, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512]]
        down_scales = [1,2,3]
        self.conv_embed_list = nn.LayerList([
            nn.Sequential(
                nn.MaxPool2D(2),
                nn.Conv2D(channel[0], channel[1], 3, 1, 1),
                Rearrange('b c h w -> b (h w) c',
                          h=image_size//(2**(i)), w=image_size//(2**(i))),
                nn.LayerNorm(channel[1])
            ) for i, channel in map(lambda i,a: (i, a), down_scales, down_channels)
        ])
        self.transformer_list = nn.LayerList([
            nn.Sequential(
            Transformer(axis=channel[1], img_size=image_size//(2**i),depth=depth, heads=heads, axis_head=channel[1],
                                              mlp_axis=channel[1] * scale_axis, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//(2**i), w = image_size//(2**i))
        ) for i, channel, depth, heads in map(lambda i, a, b, c: (i, a, b, c), down_scales, down_channels, depth, heads)
        ])
        self.post_conv_pool = nn.Sequential(
            nn.MaxPool2D(2),
            layers.ConvBNReLU(512, 512, 3)
        )

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for i in range(len(self.conv_embed_list)):
            short_cuts.append(x)
            x = self.conv_embed_list[i](x)
            x = self.transformer_list[i](x)
        short_cuts.append(x)
        x = self.post_conv_pool(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, mode='bilinear'):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, mode)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 mode):
        super().__init__()

        self.align_corners = align_corners

        self.upsample = nn.Upsample(
            scale_factor=2, mode=mode, align_corners=align_corners)
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels+in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        x = self.upsample(x)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x
