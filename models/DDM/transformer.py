# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-10-09 21:19:46
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 15:43:21
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(dim, d_model), 3)
        self.attn = None

        self.to_out = nn.Linear(d_model, dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        nbatches = x.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(y).view(nbatches, self.h, self.d_k)
             for l, y in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=None, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.view(nbatches, self.h * self.d_k)
        return self.dropout(self.to_out(x))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        x = self.sublayer[1](x, lambda x: self.self_attn(x))
        return self.sublayer[2](x, self.feed_forward)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # GHZ state: Permutation Invariant
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device

        position = torch.arange(self.dim, device=device)
        div_term = torch.exp(torch.arange(self.dim, device=device) *
                             -(math.log(10000.0) / self.dim))
        emb_sin = torch.sin(position * div_term)
        emb_cos = torch.cos(position * div_term)
        return torch.cat((x + emb_sin, x + emb_cos), dim=1)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        out = self.proj(x)
        return out

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim_out), LayerNorm(dim_out)
        )
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.block(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2), nn.Dropout(0.1))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        if self.mlp is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class Transformer(nn.Module):  # Transformer
    def __init__(self, input_dim, output_dim, num_layers=1, d_model=64, d_ff=2048, h=4, channels=32, dropout=0.05, device='cpu'):
        super(Transformer, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, channels * 4), 
            nn.SiLU(), 
            nn.Linear(channels * 4, channels)
        )

        embed_dim = input_dim * 2
        self.PositionEmb = PositionEmb(input_dim)

        self.resnet = ResnetBlock(embed_dim, embed_dim, time_emb_dim=channels)
        self.resnet2 = ResnetBlock(embed_dim, embed_dim, time_emb_dim=channels)

        c = copy.deepcopy
        attn = MultiHeadedAttention(embed_dim, h, d_model)
        ff = PositionwiseFeedForward(embed_dim, d_ff, dropout)
        self.attn = nn.Sequential(
            Decoder(DecoderLayer(embed_dim, c(attn), c(ff), dropout), num_layers),
            Generator(embed_dim, embed_dim)
            )

        for p in self.attn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.SiLU(), 
            nn.Linear(embed_dim, output_dim)
            )

        self.to(device)

    def forward(self, x, t):
        x = self.PositionEmb(x)
        t = self.time_mlp(t)

        out = self.resnet(x, t)
        out = self.attn(out)
        out = self.resnet2(out, t)

        return self.to_out(out)
