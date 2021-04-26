# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-04-26 00:25:05
Description: None
'''
import paddle
from paddleseg.models.hrunet import Encoder
from paddleseg.models.backbones import HRNet_W64

if __name__ == '__main__':
    x = paddle.ones([1, 3, 256, 256])
    model = Encoder()
    y = model(x)
    print(y[0].shape)
