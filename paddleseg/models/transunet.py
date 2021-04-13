# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-29 13:50:36
LastEditors: TJUZQC
LastEditTime: 2021-04-09 23:50:15
Description: None
'''
from typing import Iterable

import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddleseg.models.unet import Encoder
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models.layers import ConvBNReLU, PositionEmbeddingLearned
from paddle.nn import functional as F


@manager.MODELS.add_component
class TransUNet(nn.Layer):
    """
    This class implements a TransUNet like semantic segmentation model.
    """

    def __init__(self,
                 num_classes: int = 3,
                 number_of_query_positions: int = 12,
                 hidden_features=512,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 transformer_attention_heads: int = 8,
                 transformer_activation: str = 'leakyrelu',
                 segmentation_attention_heads: int = 8,
                #  segmentation_head_final_activation: str = 'sigmoid',
                 resample_mode='bilinear',) -> None:
        """
        Constructor method
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
        :param resample_mode: (str) Type of resample mode
        """
        # Call super constructor
        super(TransUNet, self).__init__()
        # Init backbone
        self.encoder = Encoder()
        # Init convolution mapping to match transformer dims
        self.convolution_mapping = ConvBNReLU(in_channels=self.encoder.down_channels[-1][-1], out_channels=hidden_features, kernel_size=3)
        # Init query positions
        self.query_positions = self.create_parameter(
            [number_of_query_positions, hidden_features], dtype='float32')
        # Init embeddings
        self.positionembedding = PositionEmbeddingLearned(hidden_features)

        # Init transformer activation
        self.transformer_activation = _get_activation(transformer_activation)

        # Init segmentation final activation
        # self.segmentation_final_activation = _get_activation(segmentation_head_final_activation)
        # self.segmentation_final_activation = self.segmentation_final_activation(axis=1) if isinstance(
        #     self.segmentation_final_activation(), nn.Softmax) else self.segmentation_final_activation()
        pass


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
        self.decoder = AttentionDecoder(align_corners=False, mode=resample_mode)
        # Init classification layer
        self.cls = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        

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
        decoded_features = self.decoder(features_encoded, feature_list)
        instance_segmentation_prediction = self.cls(decoded_features)

        return instance_segmentation_prediction

def _get_activation(activation:str):
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

class AttentionDecoder(nn.Layer):
    def __init__(self, align_corners, mode='bilinear'):
        super(AttentionDecoder, self).__init__()

        self.up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, mode=mode)
            for channel in self.up_channels
        ])
        

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 mode):
        super().__init__()

        self.align_corners = align_corners

        self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)
        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels+in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        x = self.upsample(x)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x