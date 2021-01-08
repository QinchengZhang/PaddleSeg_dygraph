# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-01-08 22:15:08
Description: None
'''
from paddle import nn, Tensor
import paddle
from paddleseg.models.layers import CVTransformerEncoderLayer as TransformerEncoderlayer
from paddleseg.models import CellSETR

if __name__ == '__main__':
    input = paddle.ones([1, 3, 256, 256])
    model = CellSETR(3, segmentation_head_final_activation=paddle.nn.Softmax)
    print(paddle.summary(model, (1, 3, 256, 256)))
    # output = model(input)
    # print(output.shape)