#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

__all__ = ['InputEmbeddings', 'PositionalEncoding']

class InputEmbeddings(nn.Module):
  def __init__(self, d_model, vocab_size):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size

    # embedding layer from Pytorch
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # section 3.3, weights multiplied by sqrt(d_model)
    return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model).float())

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, seq_len, dropout):
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # positionalencoding is of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # position vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

    # sine for even positions and cosine for odd positions
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0) # add batch dim (1, seq_len, d_model)

    # pe is not a learned parameter but it is part of the model
    # register pe so that it gets saved as part of the model for saving and loading
    self.register_buffer('pe', pe)
    
  def forward(self, x):
    # since pe is fixed and not learned, we set requires_grad to False
    x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
    return self.dropout(x)    