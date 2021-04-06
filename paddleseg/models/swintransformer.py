# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-03-31 10:27:53
LastEditors: TJUZQC
LastEditTime: 2021-04-01 12:20:28
Description: None
'''
from paddleseg.models.layers.layer_libs import ConvBN, ConvBNReLU
from typing import Iterable

import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.hsunet import Encoder
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models.layers import HSBottleNeck, PositionEmbeddingLearned, hierarchicalsplit
from paddleseg.utils.einops.layers.paddle import Rearrange
import numpy as np
from paddleseg import utils


@manager.MODELS.add_component
class SwinTransUNet(nn.Layer):
    def __init__(self,
                 n_channels: int = 3,
                 num_classes: int = 2,
                 C=1024):
        # Call super constructor
        super(SwinTransUNet, self).__init__()
        self.patch_size = 4
        self.patch_partition = Rearrange(
            'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        self.linear_embedding = nn.Linear(n_channels*(self.patch_size**2), C)

    def forward(self, x):
        b,c,h,w = x.shape
        num_of_patches = (h/self.patch_size)*(w/self.patch_size)
        out = self.patch_partition(x)
        print(out.shape)
        return self.linear_embedding(out)
