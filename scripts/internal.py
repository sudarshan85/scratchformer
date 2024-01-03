#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

__all__ = ['LayerNormalization', 'FeedForwardBlock', 'MultiHeadAttention', 'ResidualConnection']

class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # multiplictive 
    self.bias = nn.Parameter(torch.zeros(0)) # additive

  def forward(self, x) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

# section 3.3 position-wise FFN
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 & b1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 & b2

  def forward(self, x):
    # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model is not divisible by h"
    self.d_k = d_model // h
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    # token to token attention, so matrix is seq_len, seq_len
    # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
      # fill those masked location with large negative number so it softmaxes to zero
      attention_scores.masked_fill(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return (attention_scores @ value), attention_scores

  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    key = self.w_k(k)
    value = self.w_v(v)

    # 1) reshape q,k,v into separate heads
    # 2) put the head dim as the 2nd dim
    # 3) each head will see part of the embedding of ALL inputs in the batch
    # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) 

    x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
    # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    return self.w_o(x)

class ResidualConnection(nn.Module):
  def __init__(self, dropout: float):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))