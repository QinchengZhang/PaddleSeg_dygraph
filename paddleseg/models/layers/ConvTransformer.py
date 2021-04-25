# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-04-22 14:20:33
LastEditors: TJUZQC
LastEditTime: 2021-04-23 14:19:04
Description: None
'''
from typing import Optional

import numpy as np
import paddle.nn.functional as F
import paddle
from paddle import Tensor, nn
from paddleseg.models.layers import Identity
from paddleseg.utils.einops import rearrange, repeat
from paddleseg.utils.einops.layers.paddle import Rearrange


class SepConv2d(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2D(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = nn.BatchNorm2D(in_channels)
        self.pointwise = nn.Conv2D(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Layer):
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Layer):
    def __init__(self, axis, hidden_axis, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(axis, hidden_axis),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_axis, axis),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConvAttention(nn.Layer):
    def __init__(self, axis, img_size, heads = 8, axis_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_axis = axis_head *  heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(axis, inner_axis, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(axis, inner_axis, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(axis, inner_axis, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = paddle.concat((cls_token, q), axis=2)
            v = paddle.concat((cls_token, v), axis=2)
            k = paddle.concat((cls_token, k), axis=2)


        dots = paddle.to_tensor(np.einsum('b h i d, b h j d -> b h i j', q.numpy(), k.numpy()) * self.scale)

        attn = F.softmax(dots, axis=-1)

        out = paddle.to_tensor(np.einsum('b h i j, b h j d -> b h i d', attn.numpy(), v.numpy()))
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
        
class Transformer(nn.Layer):
    def __init__(self, axis, img_size, depth, heads, axis_head, mlp_axis, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(axis, ConvAttention(axis, img_size, heads=heads, axis_head=axis_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(axis, FeedForward(axis, mlp_axis, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT(nn.Layer):
    def __init__(self, image_size, in_channels, num_classes, axis=64, kernels=[3, 3, 3], strides=[1, 1, 1],
                 heads=[1, 3, 6] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_axis=4):
        super().__init__()




        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.axis = axis

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2D(in_channels, axis, kernels[0], strides[0], 1),
            nn.MaxPool2D(2),
            Rearrange('b c h w -> b (h w) c', h = image_size//2, w = image_size//2),
            nn.LayerNorm(axis)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(axis=axis, img_size=image_size//2,depth=depth[0], heads=heads[0], axis_head=self.axis,
                                              mlp_axis=axis * scale_axis, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//2, w = image_size//2)
        )

        ##### Stage 2 #######
        in_channels = axis
        scale = heads[1]//heads[0]
        axis = scale*axis
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2D(in_channels, axis, kernels[1], strides[1], 1),
            nn.MaxPool2D(2),
            Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4),
            nn.LayerNorm(axis)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(axis=axis, img_size=image_size//4, depth=depth[1], heads=heads[1], axis_head=self.axis,
                                              mlp_axis=axis * scale_axis, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)
        )

        ##### Stage 3 #######
        in_channels = axis
        scale = heads[2] // heads[1]
        axis = scale * axis
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2D(in_channels, axis, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size//16, w = image_size//16),
            nn.LayerNorm(axis)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(axis=axis, img_size=image_size//16, depth=depth[2], heads=heads[2], axis_head=self.axis,
                                              mlp_axis=axis * scale_axis, dropout=dropout, last_stage=True),
        )


        self.cls_token = self.create_parameter([1, 1, axis])
        self.dropout_large = nn.Dropout(emb_dropout)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(axis),
            nn.Linear(axis, num_classes)
        )

    def forward(self, img):
        # img.shape = 1,3,256,256
        xs = self.stage1_conv_embed(img)  #  1,3,256,256 -> 1,4096,64
        xs = self.stage1_transformer(xs)  #  1,4096,64 -> 1,64,64,64

        xs = self.stage2_conv_embed(xs)  #  1,64,64,64 -> 1,1024,192
        xs = self.stage2_transformer(xs)  #  1,1024,192 -> 1,192,32,32

        xs = self.stage3_conv_embed(xs)  #  1,192,32,32 -> 1,256,384
        b, _, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  #  1,256,384
        xs = paddle.concat((cls_tokens, xs), axis=1)  #  1,256,384 -> 1,257,384
        xs = self.stage3_transformer(xs)  # 1,257,384
        xs = xs.mean(axis=1) if self.pool == 'mean' else xs[:, 0]  # 1,257,384 -> 1,384

        xs = self.mlp_head(xs)  # 1,384 -> 1,2
        return xs