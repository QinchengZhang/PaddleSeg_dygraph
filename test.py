# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-04-23 14:24:36
Description: None
'''
from paddleseg.utils.einops.layers.paddle import Rearrange
from paddleseg.models.layers import CvT
from paddleseg.models import SwinTransUNet
from paddleseg.models.convtransunet import Encoder
from paddleseg.utils.einops import repeat
import numpy as np
import paddle
    
if __name__ == '__main__':
    x = paddle.ones([1,3,256,256])
    model = Encoder(256, 3)
    y = model(x)
    print(y)