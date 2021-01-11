# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-22 14:57:05
LastEditors: TJUZQC
LastEditTime: 2021-01-11 16:49:57
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
from .unet import UNet
from .hsunet import HS_UNet
from .hsattunet import HS_Att_UNet
from .cellsetr import CellSETR
from .transunet import TransUNet