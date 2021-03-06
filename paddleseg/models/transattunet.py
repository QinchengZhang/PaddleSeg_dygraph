# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-29 13:50:36
LastEditors: TJUZQC
LastEditTime: 2021-03-06 12:32:12
Description: None
'''
from typing import Iterable

import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.hsunet import Encoder
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models.layers import HSBottleNeck, PositionEmbeddingLearned
import numpy as np
from paddleseg import utils


@manager.MODELS.add_component
class TransAttentionUNet(nn.Layer):
    """
    This class implements a TransUNet like semantic segmentation model.
    """

    def __init__(self,
                 n_channels: int = 3,
                 num_classes: int = 2,
                 number_of_query_positions: int = 12,
                 hidden_features=1024,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 transformer_attention_heads: int = 8,
                 transformer_activation: str = 'leakyrelu',
                 segmentation_attention_heads: int = 8,
                 segmentation_head_final_activation: str = 'sigmoid') -> None:
        """
        Constructor method
        :param n_channels: (int) Channel of input image
        :param num_classes: (int) Number of classes in the dataset
        :param number_of_query_positions: (int) Number of query positions
        :param hidden_features: (int) Number of hidden features in the transformer module
        :param num_encoder_layers: (int) Number of layers in encoder part of the transformer module
        :param num_decoder_layers: (int) Number of layers in decoder part of the transformer module
        :param dropout: (float) Dropout factor used in transformer module and segmentation head
        :param transformer_attention_heads: (int) Number of attention heads in the transformer module
        :param transformer_activation: (Type) Type of activation function to be utilized in the transformer module
        :param segmentation_attention_heads: (int) Number of attention heads in the 2d multi head attention module
        :param segmentation_head_final_activation: (Type) Type of activation function to be applied to the output pred
        """
        # Call super constructor
        super(TransAttentionUNet, self).__init__()
        # Init backbone
        self.encoder = Encoder(n_channels, [64, 128, 256, 512])
        # Init convolution mapping to match transformer dims
        # self.convolution_mapping = HSBottleNeck(
        #     in_channels=512, out_channels=hidden_features)
        self.convolution_mapping = ConvBlock(1024, hidden_features)
        # Init query positions
        self.query_positions = self.create_parameter(
            [number_of_query_positions, hidden_features], dtype='float32')
        # Init embeddings
        self.positionembedding = PositionEmbeddingLearned(hidden_features)

        # Init transformer activation
        self.transformer_activation = _get_activation(transformer_activation)

        # Init segmentation final activation
        self.segmentation_final_activation = _get_activation(
            segmentation_head_final_activation)
        self.segmentation_final_activation = self.segmentation_final_activation(axis=1) if isinstance(
            self.segmentation_final_activation(), nn.Softmax) else self.segmentation_final_activation()

        # Init transformer
        self.transformer = Transformer(d_model=hidden_features, nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dropout=dropout, dim_feedforward=4 * hidden_features,
                                       activation=self.transformer_activation)

        # Init segmentation attention head
        self.segmentation_attention_head = nn.MultiHeadAttention(
            embed_dim=hidden_features,
            num_heads=segmentation_attention_heads,
            dropout=dropout)
        # Init segmentation head
        filters = np.array([64, 128, 256, 512, 1024])
        self.up5 = UpConv(ch_in=filters[4], ch_out=filters[3])
        self.att5 = AttentionBlock(
            F_g=filters[3], F_l=filters[3], F_out=filters[2])
        self.up_conv5 = ConvBlock(ch_in=filters[4], ch_out=filters[3])

        self.up4 = UpConv(ch_in=filters[3], ch_out=filters[2])
        self.att4 = AttentionBlock(
            F_g=filters[2], F_l=filters[2], F_out=filters[1])
        self.up_conv4 = ConvBlock(ch_in=filters[3], ch_out=filters[2])

        self.up3 = UpConv(ch_in=filters[2], ch_out=filters[1])
        self.att3 = AttentionBlock(
            F_g=filters[1], F_l=filters[1], F_out=filters[0])
        self.up_conv3 = ConvBlock(ch_in=filters[2], ch_out=filters[1])

        self.up2 = UpConv(ch_in=filters[1], ch_out=filters[0])
        self.att2 = AttentionBlock(
            F_g=filters[0], F_l=filters[0], F_out=filters[0] // 2)
        self.up_conv2 = ConvBlock(ch_in=filters[1], ch_out=filters[0])

        self.conv_1x1 = nn.Conv2D(
            filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Init classification layer
        self.cls = HSBottleNeck(in_channels=64, out_channels=num_classes)

    def get_parameters(self, lr_main: float = 1e-04, lr_backbone: float = 1e-05) -> Iterable:
        """
        Method returns all parameters of the model with different learning rates
        :param lr_main: (float) Leaning rate of all parameters which are not included in the backbone
        :param lr_backbone: (float) Leaning rate of the backbone parameters
        :return: (Iterable) Iterable object including the main parameters of the generator network
        """
        return [{'params': self.encoder.parameters(), 'lr': lr_backbone},
                {'params': self.convolution_mapping.parameters(), 'lr': lr_main},
                {'params': [self.row_embedding], 'lr': lr_main},
                {'params': [self.column_embedding], 'lr': lr_main},
                {'params': self.transformer.parameters(), 'lr': lr_main},
                {'params': self.bounding_box_head.parameters(), 'lr': lr_main},
                {'params': self.class_head.parameters(), 'lr': lr_main},
                {'params': self.segmentation_attention_head.parameters(),
                 'lr': lr_main},
                {'params': self.decoder.parameters(), 'lr': lr_main}]

    def get_segmentation_head_parameters(self, lr: float = 1e-05) -> Iterable:
        """
        Method returns all parameter of the segmentation head and the 2d multi head attention module
        :param lr: (float) Learning rate to be utilized
        :return: (Iterable) Iterable object including the parameters of the segmentation head
        """
        return [{'params': self.segmentation_attention_head.parameters(), 'lr': lr},
                {'params': self.segmentation_head.parameters(), 'lr': lr}]

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass
        :param input: (paddle.Tensor) Input image of shape (batch size, channels, height, width)
        :return: (Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]) Class prediction, bounding box predictions and
        segmentation maps
        """
        # Get features from backbone
        features, feature_list = self.encoder(input)
        # Map features to the desired shape
        features = self.convolution_mapping(features)
        # Make positional embeddings
        positional_embeddings = self.positionembedding(features)
        # Get transformer encoded features
        latent_tensor, features_encoded = self.transformer(
            features, self.query_positions, positional_embeddings)
        latent_tensor = latent_tensor.transpose([2, 0, 1])
        # Get instance segmentation prediction
        d5 = self.up5(features_encoded)
        x4 = self.att5(g=d5, x=feature_list[3])
        d5 = paddle.concat([x4, d5], axis=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=feature_list[2])
        d4 = paddle.concat((x3, d4), axis=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=feature_list[1])
        d3 = paddle.concat((x2, d3), axis=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=feature_list[0])
        d2 = paddle.concat((x1, d2), axis=1)
        d2 = self.up_conv2(d2)

        logit = self.conv_1x1(d2)
        # logit_list = [logit]
        # decoded_features = self.decoder(features_encoded, feature_list)
        # instance_segmentation_prediction = self.cls(decoded_features)

        return logit


def _get_activation(activation: str):
    activation = activation.lower()
    switch = {
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'hardshrink': nn.Hardshrink,
        'hardswish': nn.Hardswish,
        'tanh': nn.Tanh,
        'hardtanh': nn.Hardtanh,
        'prelu': nn.PReLU,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'selu': nn.SELU,
        'leakyrelu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'hardsigmoid': nn.Hardsigmoid,
        'softmax': nn.Softmax,
        'softplus': nn.Softplus,
        'softshrink': nn.Softshrink,
        'softsign': nn.Softsign,
        'swish': nn.Swish,
        'tanhshrink': nn.Tanhshrink,
        'thresholdedrelu': nn.ThresholdedReLU,
        'logsigmoid': nn.LogSigmoid,
        'logsoftmax': nn.LogSoftmax,
        'maxout': nn.Maxout,
    }
    return switch.get(activation, None)

class Encoder(nn.Layer):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(input_channels, 64, 3),
            layers.ConvBNReLU(64, 64, 3))
        down_channels = filters
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel, channel * 2)
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts
        
class AttentionBlock(nn.Layer):
    def __init__(self, F_g, F_l, F_out):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_out))

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_out))

        self.psi = nn.Sequential(
            nn.Conv2D(F_out, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(1), nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        res = x * psi
        return res

class UpConv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out), nn.ReLU())

    def forward(self, x):
        return self.up(x)

class AttentionDecoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False, split=5):
        super(AttentionDecoder, self).__init__()

        self.up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            AttentionUpSampling(
                channel[0], channel[1], align_corners, use_deconv, split)
            for channel in self.up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class AttentionUpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False,
                 split=5):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv

        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=self.align_corners),
                nn.Conv2D(in_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(out_channels),
                nn.ReLU()
            )

        self.attention_block = AttentionBlock(
            in_channels, in_channels, out_channels)

        self.double_conv = nn.Sequential(
            HSBottleNeck(in_channels+out_channels, out_channels, split),
            HSBottleNeck(out_channels+out_channels, out_channels, split))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        print(x.shape, short_cut.shape)
        short_cut = self.attention_block(g=x, x=short_cut)
        print(x.shape, short_cut.shape)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x

class ConvBlock(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out), nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out), nn.ReLU())

    def forward(self, x):
        return self.conv(x)