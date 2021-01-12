# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-01-12 14:33:55
Description: None
'''
from paddle import nn, Tensor
import paddle
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models import CellSETR, HS_UNet, UNet
    
if __name__ == '__main__':
    paddle.set_device('cpu')
    input = paddle.ones([1, 3, 256, 256])
    label = paddle.ones([1, 256, 256])
    losses = [paddle.nn.CrossEntropyLoss()]
    model1 = CellSETR(2, segmentation_head_final_activation='sigmoid')
    model2 = UNet(2)


    # paddle.jit.save(model, 'transformer', input_spec=[features, query_positions, position_embedding])
    # print(paddle.summary(model, (1, 3, 256, 256)))
    output = model1(input)
    # loss = loss_func(output, label)
    print(output[0].shape)