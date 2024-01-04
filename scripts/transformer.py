#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

from encoder import *
from decoder import *
from input import *
from internal import *

__all__ = ['ProjectionLayer', 'Transformer', 'build_transformer']

class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim=-1)    

class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbeddings, target_emb: InputEmbeddings, src_pe: PositionalEncoding, target_pe: PositionalEncoding, projection_layer: ProjectionLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_emb = src_emb
    self.target_emb = target_emb
    self.src_pe = src_pe
    self.target_pe = target_pe
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_emb(src)
    src = self.src_pe(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output, src_mask, target, target_mask):
    target = self.target_emb(target)
    target = self.target_pe(target)    
    return self.decoder(target, encoder_output, src_mask, target_mask)

  def project(self, x):
    return self.projection_layer(x)
      
def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
  # embed
  src_emb = InputEmbeddings(d_model, src_vocab_size)
  target_emb = InputEmbeddings(d_model, target_vocab_size)

  # positional encoding
  src_pe = PositionalEncoding(d_model, src_seq_len, dropout)
  target_pe = PositionalEncoding(d_model, target_seq_len, dropout)

  # encoder
  enc_blocks = []
  for _ in range(N):
    self_attn = MultiHeadAttention(d_model, h, dropout)
    ffb = FeedForwardBlock(d_model, d_ff, dropout)
    enc_block = EncoderBlock(self_attn, ffb, dropout)
    enc_blocks.append(enc_block)
  encoder = Encoder(nn.ModuleList(enc_blocks))

  # decoder
  dec_blocks = []
  for _ in range(N):
    self_attn = MultiHeadAttention(d_model, h, dropout)
    cross_attn = MultiHeadAttention(d_model, h, dropout)
    ffb = FeedForwardBlock(d_model, d_ff, dropout)
    dec_block = DecoderBlock(self_attn, cross_attn, ffb, dropout)
    dec_blocks.append(dec_block)
  decoder = Decoder(nn.ModuleList(dec_blocks))  

  # projection
  projection = ProjectionLayer(d_model, target_vocab_size)

  transformer = Transformer(encoder, decoder, src_emb, target_emb, src_pe, target_pe, projection)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  
  return transformer