# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-04-22 15:48:29
Description: None
'''
from paddleseg.utils.einops.layers.paddle import Rearrange
from paddleseg.models.layers import CvT
from paddleseg.models import SwinTransUNet
from paddleseg.models.unet import Encoder
from paddleseg.utils.einops import repeat
import numpy as np
import paddle
    
if __name__ == '__main__':
    x = paddle.ones([1,3,256,256])
    model = Encoder()
    y = model(x)
    print(y)