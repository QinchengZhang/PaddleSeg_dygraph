# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-29 13:50:36
LastEditors: TJUZQC
LastEditTime: 2021-01-08 22:22:49
Description: None
'''
from typing import Iterable, Tuple, Type

import paddle
import paddle.nn as nn
from paddleseg.models.hsunet import Decoder, Encoder
from paddleseg.models.layers import CVTransformer as Transformer
from paddleseg.models.layers import HSBottleNeck, PositionEmbeddingLearned


class CellSETR(nn.Layer):
    """
    This class implements a DETR (Facebook AI) like semantic segmentation model.
    """

    def __init__(self,
                 num_classes: int = 3,
                 number_of_query_positions: int = 12,
                 hidden_features=512,
                 backbone_channels: Tuple[Tuple[int, int], ...] = (
                     (1, 64), (64, 128), (128, 256), (256, 256), (256, 512), (512, 512)),
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.0,
                 transformer_attention_heads: int = 8,
                 transformer_activation: Type = nn.LeakyReLU,
                 segmentation_attention_heads: int = 8,
                 segmentation_head_final_activation: Type = nn.Sigmoid) -> None:
        """
        Constructor method
        :param num_classes: (int) Number of classes in the dataset
        :param number_of_query_positions: (int) Number of query positions
        :param hidden_features: (int) Number of hidden features in the transformer module
        :param backbone_channels: (Tuple[Tuple[int, int], ...]) In and output channels of each block in the backbone
        :param num_encoder_layers: (int) Number of layers in encoder part of the transformer module
        :param num_decoder_layers: (int) Number of layers in decoder part of the transformer module
        :param dropout: (float) Dropout factor used in transformer module and segmentation head
        :param transformer_attention_heads: (int) Number of attention heads in the transformer module
        :param transformer_activation: (Type) Type of activation function to be utilized in the transformer module
        :param segmentation_attention_heads: (int) Number of attention heads in the 2d multi head attention module
        :param segmentation_head_final_activation: (Type) Type of activation function to be applied to the output pred
        """
        # Call super constructor
        super(CellSETR, self).__init__()
        # Init backbone
        self.encoder = Encoder(split=5)
        # Init convolution mapping to match transformer dims
        self.convolution_mapping = nn.Conv2D(in_channels=backbone_channels[-1][-1], out_channels=hidden_features,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias_attr=True)
        # Init query positions
        self.query_positions = self.create_parameter(
            [number_of_query_positions, hidden_features], dtype='float32')
        # Init embeddings
        self.positionembedding = PositionEmbeddingLearned(hidden_features)

        # Init transformer
        self.transformer = Transformer(d_model=hidden_features, nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dropout=dropout, dim_feedforward=4 * hidden_features,
                                       activation=transformer_activation)

        # Init segmentation attention head
        self.segmentation_attention_head = nn.MultiHeadAttention(
            embed_dim=hidden_features,
            num_heads=segmentation_attention_heads,
            dropout=dropout)
        # Init segmentation head
        self.decoder = Decoder(align_corners=False, use_deconv=True, split=5)
        # Init classification layer
        self.cls = HSBottleNeck(in_channels=64, out_channels=num_classes)
        # Init final segmentation activation
        self.segmentation_final_activation = segmentation_head_final_activation(axis=1) if isinstance(
            segmentation_head_final_activation(), nn.Softmax) else segmentation_head_final_activation()

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

        return self.segmentation_final_activation(instance_segmentation_prediction)
