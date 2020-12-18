# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-03 14:31:32
LastEditors: TJUZQC
LastEditTime: 2020-12-18 14:11:03
Description: None
'''
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.
    """

    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, input, label):
        """
        Forward computation.

        Args:
            input (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if input.shape[1] == 1:
            input = paddle.squeeze(input, 1)
        else:
            input = paddle.argmax(input, axis=1).astype('float32')
        assert len(label.shape) == len(
            input.shape), f'The shape of label must be (N, D1, D2,..., Dk) but got (N, {label.shape[1]}, D1, D2,..., Dk)'
        label[label == self.ignore_index] = 0
        out = self._compute(input, label.astype(input.dtype), self.EPS)
        label.stop_gradient = True
        return out

    def _compute(self, input, label, eps=1e-5):
        inse = paddle.sum(input * label)
        dice_denominator = paddle.sum(input**2 + label**2)
        dice_score = 1. - (inse * 2) / (dice_denominator + eps)
        return paddle.mean(dice_score)
