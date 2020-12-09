# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-03 14:31:32
LastEditors: TJUZQC
LastEditTime: 2020-12-09 15:00:59
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

    def __init__(self):
        super(DiceLoss, self).__init__()
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
        label = paddle.unsqueeze(label, 1)
        assert len(label.shape) == len(input.shape), 'The shape of logit and label must be same'
        perm = [i for i in range(len(label.shape))]
        perm.remove(1)
        perm.append(1)

        input = paddle.transpose(input, perm)
        label = paddle.transpose(label, perm)
        out = F.dice_loss(
            input, label, self.EPS)
        label.stop_gradient = True
        return out
