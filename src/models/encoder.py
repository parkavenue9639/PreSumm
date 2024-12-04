import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer as PyTorchTransformerEncoderLayer


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000, apply_dropout=True):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.apply_dropout = apply_dropout

        # 构建位置编码矩阵
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # 扩展 batch 维度
        self.register_buffer('pe', pe)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, step=None):
        """
        添加位置编码到输入 emb 上
        :param emb: 输入嵌入，形状 [batch_size, seq_len, dim]
        :param step: 如果指定，仅添加某一时刻的位置编码
        """
        if step is not None:
            emb = emb + self.pe[:, step].unsqueeze(1)  # 单步
        else:
            emb = emb + self.pe[:, :emb.size(1)]  # 整体
        if self.apply_dropout:
            emb = self.dropout(emb)
        return emb

    def get_emb(self, n_sents=None, step=None):
        """
        获取位置编码
        :param n_sents: 如果指定，返回前 n_sents 的位置编码
        :param step: 如果指定，仅返回 step 时刻的编码
        """
        if step is not None:
            return self.pe[:, step].unsqueeze(1)  # [1, 1, dim]
        if n_sents is not None:
            return self.pe[:, :n_sents]  # [1, n_sents, dim]
        return self.pe



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        """
        Args:
            d_model: 输入和输出的特征维度，决定了每个句子表示的嵌入维度
            d_ff: 前馈网络隐藏层的维度，控制transformer的容量
            heads: 多头注意力机制的头数，用于捕获不同子空间的句子间关系
            dropout: 用于防止过拟合，应用在多处
            num_inter_layers: transformer的编码层数
        """
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        # 为句子添加位置嵌入，保留序列顺序信息。在序列位置上叠加正弦波嵌入，位置固定。
        self.pos_emb = PositionalEncoding(dropout=0.1, dim=d_model, max_len=5000)
        # 每层由自注意力和前馈网络组成，用于捕获句子间关系。使用pytorch的ModuleList组织多层transformer
        '''self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])'''
        # 使用堆叠的nn.TransformerEncoder，其内部实现对多层transformer进行了优化
        encoder_layer = PyTorchTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_inter = TransformerEncoder(encoder_layer, num_layers=num_inter_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # 输出层，全连接网络，将特征维度映射到单个标量（即句子重要性分数）
        self.wo = nn.Linear(d_model, 1, bias=True)
        # 将转换到[0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        # 预处理掩码
        mask = mask.float()
        expanded_mask = mask.unsqueeze(-1)

        pos_emb = self.pos_emb.get_emb(n_sents=top_vecs.size(1))

        # 忽略填充部分 + 叠加位置编码
        x = top_vecs * expanded_mask + pos_emb

        # 多层注意力
        src_key_padding_mask = ~mask.bool()
        for _ in range(self.num_inter_layers):
            x = x.transpose(0, 1)  # 或优化为直接支持
            x = self.transformer_inter(x, src_key_padding_mask=src_key_padding_mask)
            x = x.transpose(0, 1)

        # 层归一化
        x = self.layer_norm(x)

        # 线性层和掩码过滤
        sent_scores = self.sigmoid(self.wo(x)) * expanded_mask
        return sent_scores.squeeze(-1)


