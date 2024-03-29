# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-03-04 09:37:34
LastEditors: TJUZQC
LastEditTime: 2021-04-26 00:19:27
Description: None
'''
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .backbones import *
from .losses import *

from .ann import *
from .bisenet import *
from .danet import *
from .deeplab import *
from .fast_scnn import *
from .fcn import *
from .gcnet import *
from .ocrnet import *
from .pspnet import *
from .gscnn import GSCNN
from .unet import *
from .hardnet import HarDNet
from .u2net import U2Net, U2Netp
from .attention_unet import AttentionUNet
from .unet_plusplus import UNetPlusPlus
from .emanet import *
from .isanet import *
from .hsattunet import HSAttentionUNet
from .hsunet import HS_UNet
from .transunet import TransUNet
from .transattunet import TransAttentionUNet
from .ltunet import SwinTransUNet
from .attunetwithaspp import ASPPAttentionUNet
from .convtransunet import ConvTransUNet
from .hrunet import *
from .hs_hrunet import *
from .hrunet_maxpool import *