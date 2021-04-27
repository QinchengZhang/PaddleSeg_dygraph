# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-04-27 22:37:34
Description: None
'''
import sys   
sys.setrecursionlimit(100000) #例如这里设置为十万  
from paddleseg.models import HRUNet
from paddleseg.models.backbones import ResNet50_vd
import paddle
import os

if __name__ == '__main__':
    # input = paddle.ones([1, 3, 256, 256])
    model = HRUNet(2)
    params = paddle.summary(model, input_size=(1,3,256,256), dtypes='float32')
    # print(params)
