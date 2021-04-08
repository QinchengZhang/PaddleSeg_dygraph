# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-04-06 16:34:53
Description: None
'''
from paddleseg.utils.einops.layers.paddle import Rearrange
from paddleseg.models.layers import SwinTransformer,swin_t
from paddleseg.models import SwinTransUNet
import paddle
paddle.set_device('cpu')
    
if __name__ == '__main__':
    a = paddle.ones([1,3,256,256])
    gt = paddle.ones([1,1,256,256])
    model = SwinTransUNet(hidden_dim=128, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24))
    b = model(a)
    lossfunc = paddle.nn.CrossEntropyLoss()
    loss = lossfunc(b[0],gt)
    loss.backward()