# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-01-08 15:25:15
LastEditors: TJUZQC
LastEditTime: 2021-01-12 14:10:57
Description: None
'''
from paddle import nn, Tensor
import paddle
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models import CellSETR, HS_UNet

def check_logits_losses(logits, losses):
    len_logits = len(logits)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits, label, losses):
    check_logits_losses(logits, losses)
    loss = 0
    for i in range(len(logits)):
        logit = logits[i]
        if logit.shape[-2:] != label.shape[-2:]:
            logit = F.interpolate(
                logit,
                label.shape[-2:],
                mode='bilinear',
                align_corners=True,
                align_mode=1)
        loss_i = losses['types'][i](logit, label)
        loss += losses['coef'][i] * loss_i
    return loss
    
if __name__ == '__main__':
    paddle.set_device('cpu')
    input = paddle.ones([1, 3, 256, 256])
    label = paddle.ones([1, 256, 256])
    losses = [paddle.nn.CrossEntropyLoss()]
    model1 = CellSETR(3, segmentation_head_final_activation='Softmax')
    model2 = HS_UNet(3)


    # paddle.jit.save(model, 'transformer', input_spec=[features, query_positions, position_embedding])
    # print(paddle.summary(model, (1, 3, 256, 256)))
    output = model2(input)
    # loss = loss_func(output, label)
    print(output.shape)