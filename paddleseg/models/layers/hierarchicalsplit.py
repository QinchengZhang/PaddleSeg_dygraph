# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-12 15:26:14
LastEditors: TJUZQC
LastEditTime: 2021-01-12 15:27:37
Description: None
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.layers import SyncBatchNorm, ConvBNReLU, ConvBN

class HSBlock(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock, self).__init__()
        self.w = w
        self.channels = w*split
        self.split = split
        self.stride = stride
        self.ops_list = nn.LayerList()
        for s in range(1, self.split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            self.ops_list.append(nn.Sequential(
                nn.Conv2D(hc, hc, kernel_size=3,
                          padding=1, stride=self.stride),
                # nn.BatchNorm2D(hc),
                # nn.ReLU(),
            ))

    def forward(self, x):
        last_split = None
        channels = x.shape[1]
        assert channels == self.w * \
            self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        retfeature = x[:, 0:self.w, :, :]
        for s in range(1, self.split):
            temp = x[:, s*self.w:(s+1)*self.w, :, :] if last_split is None else paddle.concat(
                [last_split, x[:, s*self.w:(s+1)*self.w, :, :]], axis=1)
            temp = self.ops_list[s-1](temp)
            x1, x2 = self._split(temp)
            retfeature = paddle.concat([retfeature, x1], axis=1)
            last_split = x2
        retfeature = paddle.concat([retfeature, last_split], axis=1)
        del last_split
        return retfeature

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]


class HSBlockBNReLU(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlockBNReLU, self).__init__()
        self._hsblock = HSBlock(w, split, stride)
        self._batch_norm = SyncBatchNorm(split*w)

    def forward(self, x):
        x = self._hsblock(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class HSBottleNeck(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, split: int = 5, stride: int = 1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            ConvBNReLU(in_channels, self.w*split,
                       kernel_size=1, stride=stride),
            HSBlockBNReLU(self.w, split, stride),
            ConvBN(self.w*split, out_channels,
                   kernel_size=1, stride=stride),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(
                in_channels, out_channels, stride=stride, kernel_size=1)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return F.relu(residual + shortcut)