# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-01-11 12:49:53
Description: None
'''
from paddle import nn, Tensor
import paddle
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models import CellSETR

if __name__ == '__main__':
    input = paddle.ones([1, 3, 256, 256])
    features = paddle.ones([1, 512, 16, 16])
    query_positions = paddle.ones([12, 512])
    position_embedding = paddle.ones([1, 512, 16, 16])
    # model = CellSETR(3, segmentation_head_final_activation='Softmax')
    model = Transformer()

    paddle.jit.save(model, 'transformer', input_spec=[features, query_positions, position_embedding])
    # print(paddle.summary(model, (1, 3, 256, 256)))
    # output = model(input)
    # print(output.shape)