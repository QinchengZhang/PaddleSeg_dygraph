# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-02-06 19:26:59
LastEditors: TJUZQC
LastEditTime: 2021-04-08 13:43:24
Description: None
'''
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models.losses import CrossEntropyLoss, DiceLoss


@manager.LOSSES.add_component
class CE_DSC_Loss(nn.Layer):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(CE_DSC_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5
        self.diceloss = DiceLoss(ignore_index)
        self.celoss = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        return self.diceloss(logits, labels) + self.celoss(logits, labels)