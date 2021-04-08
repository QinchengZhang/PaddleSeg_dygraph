# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2021-04-06 09:54:12
LastEditors: TJUZQC
LastEditTime: 2021-04-06 14:59:37
Description: None
'''
from typing import Optional

import numpy as np
import paddle
from paddle import Tensor, nn
from paddle.fluid.layers.nn import shape
from paddleseg.utils.einops import rearrange
from paddleseg.utils.einops.layers.paddle import Rearrange


class CyclicShift(nn.Layer):
    def __init__(self, displacement):
        super(CyclicShift, self).__init__()
        self.displacement = displacement
    
    def forward(self, x):
        return paddle.roll(x, shifts=(self.displacement, self.displacement), axis=(1,2))

class Residual(nn.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn        

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = paddle.zeros([window_size ** 2, window_size ** 2])

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    distances = paddle.to_tensor(indices[np.newaxis, :, :] - indices[:, np.newaxis, :])
    return distances

class WindowAttention(nn.Layer):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.softmax = nn.Softmax()

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False)
            self.left_right_mask = create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = paddle.create_parameter(shape=[2 * window_size - 1, 2 * window_size - 1], dtype='float32')
            # self.pos_embedding = nn.Parameter(paddle.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = paddle.create_parameter(shape=[window_size ** 2, window_size ** 2], dtype='float32')
            # self.pos_embedding = nn.Parameter()

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = paddle.to_tensor(np.einsum('b h w i d, b h w j d -> b h w i j', q.numpy(), k.numpy())) * self.scale
        if self.relative_pos_embedding:
            dots += paddle.to_tensor(self.pos_embedding.numpy()[self.relative_indices[:, :, 0].numpy(), self.relative_indices[:, :, 1].numpy()])
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        attn = self.softmax(dots)

        out = paddle.to_tensor(np.einsum('b h w i j, b h w j d -> b h w i d', attn.numpy(), v.numpy()))
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class SwinBlock(nn.Layer):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Layer):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = nn.functional.unfold(x, kernel_sizes=self.downscaling_factor, strides=self.downscaling_factor, paddings=0)
        x = x.reshape([b, -1, new_h, new_w]).transpose([0, 2, 3, 1])
        x = self.linear(x)
        return x

class StageModule(nn.Layer):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.LayerList([])
        for _ in range(layers // 2):
            self.layers.append(nn.LayerList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.transpose([0, 3, 1, 2])

class SwinTransformer(nn.Layer):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=8,
                 downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

    def forward(self, img):
        short_cuts = []
        print("input size:", img.shape)
        x = self.stage1(img)
        short_cuts.append(x)
        print("after stage1 size:", x.shape)
        x = self.stage2(x)
        short_cuts.append(x)
        print("after stage2 size:", x.shape)
        x = self.stage3(x)
        short_cuts.append(x)
        print("after stage3 size:", x.shape)
        x = self.stage4(x)
        print("after stage4 size:", x.shape)
        return x, short_cuts
    
def swin_t(hidden_dim=128, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)