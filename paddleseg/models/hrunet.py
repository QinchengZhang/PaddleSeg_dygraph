# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-04-25 23:40:02
LastEditors: TJUZQC
LastEditTime: 2021-04-26 18:23:01
Description: None
'''
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from re import S
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional.common import upsample
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.backbones.hrnet import Layer1, TransitionLayer, Stage
from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class HRUNet(nn.Layer):
    """
    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 resample_mode='bilinear',
                 pretrained=None):
        super().__init__()

        self.encode = Encoder()
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
    def __init__(self,
                 pretrained=None,
                stage1_num_modules=1,
                stage1_num_blocks=[4],
                stage1_num_channels=[64],
                stage2_num_modules=1,
                stage2_num_blocks=[4, 4],
                stage2_num_channels=[64, 128],
                stage3_num_modules=4,
                stage3_num_blocks=[4, 4, 4],
                stage3_num_channels=[64, 128, 256],
                stage4_num_modules=3,
                stage4_num_blocks=[4, 4, 4, 4],
                stage4_num_channels=[64, 128, 256, 512],
                 has_se=False,
                 align_corners=False):
        super(Encoder, self).__init__()
        self.pretrained = pretrained
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [sum(stage4_num_channels)]

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same',
            bias_attr=False
            ),
            layers.ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same',
            bias_attr=False
            )
        )

        self.post_double_conv = nn.Sequential(
            layers.ConvBNReLU(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding='same',
            bias_attr=False
            )
        )

        self.conv_layer1_1 = layers.ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same',
            bias_attr=False)

        self.conv_layer1_2 = layers.ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same',
            bias_attr=False)

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],
            num_filters=self.stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],
            out_channels=self.stage2_num_channels,
            name="tr1")

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,
            num_modules=self.stage2_num_modules,
            num_blocks=self.stage2_num_blocks,
            num_filters=self.stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners)

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,
            out_channels=self.stage3_num_channels,
            name="tr2")
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,
            num_modules=self.stage3_num_modules,
            num_blocks=self.stage3_num_blocks,
            num_filters=self.stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners)

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,
            out_channels=self.stage4_num_channels,
            name="tr3")
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,
            num_modules=self.stage4_num_modules,
            num_blocks=self.stage4_num_blocks,
            num_filters=self.stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners)
        self.init_weight()

    def forward(self, x): # 1 3 256 256
        # conv1 = self.conv_layer1_1(x) # 1 64 128 128
        # conv2 = self.conv_layer1_2(conv1) # 1 64 64 64
        conv1 = self.double_conv(x)


        la1 = self.la1(conv1) # 1 256 64 64

        tr1 = self.tr1([la1]) # [[1 64 64 64], [1 128 32 32]]
        st2 = self.st2(tr1) # [[1 64 64 64], [1 128 32 32]]

        tr2 = self.tr2(st2) # [[1 64 64 64], [1 128 32 32], [1 256 16 16]]
        st3 = self.st3(tr2) # [[1 64 64 64], [1 128 32 32], [1 256 16 16]]

        tr3 = self.tr3(st3) # [[1 64 64 64], [1 128 32 32], [1 256 16 16], [1 512 8 8]]
        st4 = self.st4(tr3) # [[1 64 64 64], [1 128 32 32], [1 256 16 16], [1 512 8 8]]

        out = self.post_double_conv(st4[-1])

        # x0_h, x0_w = st4[0].shape[2:]
        # x1 = F.interpolate(
        #     st4[1], (x0_h, x0_w),
        #     mode='bilinear',
        #     align_corners=self.align_corners) # b 128 64 64
        # x2 = F.interpolate(
        #     st4[2], (x0_h, x0_w),
        #     mode='bilinear',
        #     align_corners=self.align_corners) # b 256 64 64
        # x3 = F.interpolate(
        #     st4[3], (x0_h, x0_w),
        #     mode='bilinear',
        #     align_corners=self.align_corners) # b 512 64 64
        
        # x = paddle.concat([st4[0], x1, x2, x3], axis=1)

        return out, st4

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


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

        self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels+in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        x = self.upsample(x)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x
