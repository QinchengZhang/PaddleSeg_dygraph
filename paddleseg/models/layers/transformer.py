# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-29 12:36:25
LastEditors: TJUZQC
LastEditTime: 2021-01-12 15:28:07
Description: None
'''
from typing import Optional

import paddle
from paddle import Tensor, nn

class PositionEmbeddingLearned(nn.Layer):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        row_embed_weight_attr = nn.initializer.Uniform()
        col_embed_weight_attr = nn.initializer.Uniform()
        self.row_embed = nn.Embedding(
            50, num_pos_feats//2, weight_attr=row_embed_weight_attr)
        self.col_embed = nn.Embedding(
            50, num_pos_feats//2, weight_attr=col_embed_weight_attr)

    def forward(self, x):
        b, _, h, w = x.shape
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        x_emb = x_emb.unsqueeze(0)
        y_emb = y_emb.unsqueeze(1)
        x_emb = x_emb.expand([h, x_emb.shape[1], x_emb.shape[2]])
        y_emb = y_emb.expand([y_emb.shape[0], w, y_emb.shape[2]])
        pos = paddle.concat([x_emb, y_emb, ], axis=-
                            1).transpose([2, 0, 1]).unsqueeze(0)
        pos = pos.expand([b, *pos.shape[1:]])
        return pos


class CVTransformer(nn.Layer):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False,
                 return_intermediate_dec=False, initializer=nn.initializer.KaimingNormal()):
        super(CVTransformer, self).__init__()
        encoder_layer_params = (
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, initializer)
        decoder_layer_params = (
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, initializer)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = CVTransformerEncoder(
            encoder_layer_params, num_encoder_layers, encoder_norm)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = CVTransformerDecoder(decoder_layer_params, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).transpose([0, 2, 1])
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        query_embed = query_embed.unsqueeze(0).expand([bs, -1, -1])

        tgt = paddle.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose([0, 2, 1])
        memory = memory.transpose([0, 2, 1]).reshape([bs, c, h, w])
        return hs, memory


class CVTransformerEncoder(nn.Layer):

    def __init__(self, encoder_layer_params, num_layers, norm=None):
        super(CVTransformerEncoder, self).__init__()
        self.layers = nn.LayerList([CVTransformerEncoderLayer(
            *encoder_layer_params) for n in range(num_layers)])
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CVTransformerDecoder(nn.Layer):

    def __init__(self, decoder_layer_params, num_layers, norm=None, return_intermediate=False):
        super(CVTransformerDecoder, self).__init__()
        self.layers = nn.LayerList([CVTransformerDecoderLayer(
            *decoder_layer_params) for n in range(num_layers)])
        # self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output


class CVTransformerEncoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False, initializer=nn.initializer.KaimingNormal()):
        super(CVTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout, weight_attr=initializer, bias_attr=initializer)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=initializer, bias_attr=initializer)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=initializer, bias_attr=initializer)

        self.norm1 = nn.LayerNorm(d_model, weight_attr=initializer, bias_attr=initializer)
        self.norm2 = nn.LayerNorm(d_model, weight_attr=initializer, bias_attr=initializer)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Tensor):
        return tensor if pos is None else (tensor + pos)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos)
        return self.forward_post(src, src_mask, pos)


class CVTransformerDecoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU, normalize_before=False, initializer=nn.initializer.KaimingNormal()):
        super(CVTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout, weight_attr=initializer, bias_attr=initializer)
        self.multihead_attn = nn.MultiHeadAttention(
            d_model, nhead, dropout=dropout, weight_attr=initializer)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=initializer, bias_attr=initializer)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=initializer, bias_attr=initializer)

        self.norm1 = nn.LayerNorm(d_model, weight_attr=initializer, bias_attr=initializer)
        self.norm2 = nn.LayerNorm(d_model, weight_attr=initializer, bias_attr=initializer)
        self.norm3 = nn.LayerNorm(d_model, weight_attr=initializer, bias_attr=initializer)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        query = self.with_pos_embed(tgt, query_pos)
        key = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(query=query,
                                   key=key,
                                   value=memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 pos, query_pos)


def build_cvtransformer(args):
    return CVTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        initializer=args.initializer,
    )
