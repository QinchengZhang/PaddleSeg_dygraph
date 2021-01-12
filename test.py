# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-01-12 23:31:29
Description: None
'''
from paddle import nn, Tensor
import paddle
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models import CellSETR, HS_UNet, UNet
from paddleseg.datasets import BJSCLC, TN_SCUI2020
from paddleseg.transforms import Resize
    
if __name__ == '__main__':
    # dataset = TN_SCUI2020('F:/DATASET/TN-SCUI2020/segmentation/augtrain', [Resize(target_size=(256, 256))], mode='val')
    dataset = BJSCLC('F:/DATASET/Beijing-small_cell_lung_cancer-pathology/patch_1024', [Resize(target_size=(256, 256))], mode='val')
    loader = paddle.io.DataLoader(dataset)
    i = 1
    for data in loader:
        print(i, data)
        i+=1
    print('done')

    # paddle.set_device('cpu')
    # input = paddle.ones([1, 3, 256, 256])
    # label = paddle.ones([1, 256, 256])
    # losses = [paddle.nn.CrossEntropyLoss()]
    # model1 = CellSETR(2, segmentation_head_final_activation='sigmoid')
    # model2 = UNet(2)


    # # paddle.jit.save(model, 'transformer', input_spec=[features, query_positions, position_embedding])
    # # print(paddle.summary(model, (1, 3, 256, 256)))
    # output = model1(input)
    # # loss = loss_func(output, label)
    # print(output[0].shape)