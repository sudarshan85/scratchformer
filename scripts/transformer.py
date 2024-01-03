#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

__all__ = ['ProjectionLayer', 'Transformer', 'build_transformer']

class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
  def __init__(self, enocder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, target_embed: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.target_embed = target_embed
    self.src_pos = src_pos
    self.target_pos = target_pos
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_embd(src)
    src = self.src_pos(src)
    return self.encode(src, src_mask)

  def decode(self, encoder_output, src_mask, target, target_mask):
    target = self.target_embd(target)
    target = self.target_pos(target)    
    return self.decode(target, encoder_output, src_mask, target_mask)

  def project(self, x):
    return self.projection_layer(x) 

def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
  # embedding layers
  src_embed = InputEmbeddings(d_model, src_vocab_size)
  target_embed = InputEmbeddings(d_model, target_vocab_size)
  # positional encoding layers
  src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  target_pos = PositionalEncoding(d_model, target_seq_len, dropout)
  # encoder blocks
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)
    # decoder blocks
  decoder_blocks = []
  for _ in range(N):
    decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedFowardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)

  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))
  projection_layer = ProjectionLayer(d_model, target_vocab_size)

  transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer