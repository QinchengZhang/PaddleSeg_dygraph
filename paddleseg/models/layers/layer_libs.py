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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm instead"""
    if paddle.get_device() == 'cpu':
        return nn.BatchNorm(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(
            in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class HSBlock(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock, self).__init__()
        self.w = w
        self.channels = w*split
        self.split = split
        self.stride = stride
        self.ops_list = nn.LayerList()
        for s in range(1, self.split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            self.ops_list.append(nn.Sequential(
                nn.Conv2D(hc, hc, kernel_size=3, padding=1, stride=self.stride),
                # nn.BatchNorm2D(hc),
                # nn.ReLU(),
                ))

    def forward(self, x):
        last_split = None
        channels = x.shape[1]
        assert channels == self.w * \
            self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        retfeature = x[:, 0:self.w, :, :]
        for s in range(1, self.split):
            temp = x[:, s*self.w:(s+1)*self.w, :, :] if last_split is None else paddle.concat([last_split, x[:, s*self.w:(s+1)*self.w, :, :]], axis=1)
            temp = self.ops_list[s-1](temp)
            x1, x2 = self._split(temp)
            retfeature = paddle.concat([retfeature, x1], axis=1)
            last_split = x2
        retfeature = paddle.concat([retfeature, last_split], axis=1)
        del last_split
        return retfeature

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]

class HSBlockBNReLU(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlockBNReLU, self).__init__()
        self._hsblock = HSBlock(w, split, stride)
        self._batch_norm = SyncBatchNorm(split*w)

    def forward(self, x):
        x = self._hsblock(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class HSBottleNeck(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, split: int=5, stride: int = 1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            ConvBNReLU(in_channels, self.w*split, kernel_size=1, stride=stride),
            HSBlockBNReLU(self.w, split, stride),
            ConvBN(self.w*split, out_channels,
                             kernel_size=1, stride=stride),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, stride=stride, kernel_size=1)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return F.relu(residual + shortcut)

class AttentionBlock(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            ConvBN(F_g, F_int, kernel_size=1, padding=0)
        )

        self.W_x = nn.Sequential(
            ConvBN(F_l, F_int, kernel_size=1, padding=0)
        )

        self.psi = nn.Sequential(
            ConvBN(F_int, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

# class MultiHeadAttention(nn.Layer):
#     """
#     This class implements a multi head attention module like proposed in:
#     https://arxiv.org/abs/2005.12872
#     """

#     def __init__(self, query_dimension: int = 64, hidden_features: int = 64, number_of_heads: int = 16,
#                  dropout: float = 0.0) -> None:
#         """
#         Constructor method
#         :param query_dimension: (int) Dimension of query tensor
#         :param hidden_features: (int) Number of hidden features in detr
#         :param number_of_heads: (int) Number of prediction heads
#         :param dropout: (float) Dropout factor to be utilized
#         """
#         # Call super constructor
#         super(MultiHeadAttention, self).__init__()
#         # Save parameters
#         self.hidden_features = hidden_features
#         self.number_of_heads = number_of_heads
#         self.dropout = dropout
#         # Init layer
#         self.layer_box_embedding = nn.Linear(in_features=query_dimension, out_features=hidden_features, bias_attr=True)
#         # Init convolution layer
#         self.layer_image_encoding = nn.Conv2D(in_channels=query_dimension, out_channels=hidden_features,
#                                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias_attr=True)
#         # Init normalization factor
#         self.normalization_factor = paddle.to_tensor(self.hidden_features / self.number_of_heads, dtype='float32').sqrt()

#     def forward(self, input_box_embeddings: paddle.Tensor, input_image_encoding: paddle.Tensor) -> paddle.Tensor:
#         """
#         Forward pass
#         :param input_box_embeddings: (torch.Tensor) Bounding box embeddings
#         :param input_image_encoding: (torch.Tensor) Encoded image of the transformer encoder
#         :return: (torch.Tensor) Attention maps of shape (batch size, n, m, height, width)
#         """
#         # Map box embeddings
#         output_box_embeddings = self.layer_box_embedding(input_box_embeddings)
#         # Map image features
#         output_image_encoding = self.layer_image_encoding(input_image_encoding)
#         # Reshape output box embeddings
#         output_box_embeddings = output_box_embeddings.reshape([output_box_embeddings.shape[0],
#                                                            output_box_embeddings.shape[1],
#                                                            self.number_of_heads,
#                                                            self.hidden_features // self.number_of_heads])
#         # Reshape output image encoding
#         output_image_encoding = output_image_encoding.reshape([output_image_encoding.shape[0],
#                                                            self.number_of_heads,
#                                                            self.hidden_features // self.number_of_heads,
#                                                            output_image_encoding.shape[-2],
#                                                            output_image_encoding.shape[-1]])
#         # Combine tensors and normalize
#         output = np.einsum("bqnc,bnchw->bqnhw",
#                               output_box_embeddings * self.normalization_factor,
#                               output_image_encoding)
#         # Apply softmax
#         output = F.softmax(output.flatten(start_dim=2), dim=-1).view_as(output)
#         # Perform dropout if utilized
#         if self.dropout > 0.0:
#             output = F.dropout(input=output, p=self.dropout, training=self.training)
#         return output.contiguous()

class PositionEmbeddingLearned(nn.Layer):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        row_embed_weight_attr = nn.initializer.Uniform()
        col_embed_weight_attr = nn.initializer.Uniform()
        self.row_embed = nn.Embedding(50, num_pos_feats, weight_attr=row_embed_weight_attr)
        self.col_embed = nn.Embedding(50, num_pos_feats, weight_attr=col_embed_weight_attr)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = x.shape[-2:]
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        x_emb = x_emb.unsqueeze(0)
        y_emb = y_emb.unsqueeze(1)
        pos = paddle.concat([
            x_emb.expand([h, x_emb.shape[1], x_emb.shape[2]]),
            y_emb.expand([y_emb.shape[0], w, y_emb.shape[2]]),
        ], axis=-1).transpose(2, 0, 1).unsqueeze(0)
        pos = pos.expand([x.shape[0], *pos.shape[1:]])
        return pos