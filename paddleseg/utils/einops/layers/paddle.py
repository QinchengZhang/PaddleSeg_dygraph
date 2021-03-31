# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-03-30 16:59:02
LastEditors: TJUZQC
LastEditTime: 2021-03-30 17:21:12
Description: None
'''
import paddle
import numpy as np

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'TJUZQC'


class Rearrange(RearrangeMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class WeightedEinsum(WeightedEinsumMixin, paddle.nn.Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = paddle.nn.Parameter(paddle.zeros(weight_shape).uniform_(-weight_bound, weight_bound),
                                         requires_grad=True)
        if bias_shape is not None:
            self.bias = paddle.nn.Parameter(paddle.zeros(bias_shape).uniform_(-bias_bound, bias_bound),
                                           requires_grad=True)
        else:
            self.bias = None

    def forward(self, input):
        result = paddle.to_tensor(np.einsum(self.einsum_pattern, input.numpy(), self.weight.numpy()))
        if self.bias is not None:
            result += self.bias
        return result
