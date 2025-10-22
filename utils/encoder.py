import math
import torch
from torch import nn, Tensor
from multi_head_attention_mechanism import MultiHeadAttention
from position_wise_fnn import PositionWiseFNN


"""
Methods:
    forward(): tbd
Attributes:
    d_model: input embedding size.
    num_heads: number of attention heads for MHA. 
    d_ff: inner dim of position-wise feed-forward network
    activation: act-function for FNN.


"""
class EncoderBlock(nn.Module):
    

    def __init__(self, d_model, num_heads, d_ff, attn_dropout=0.0, resid_dropout=0.0, activation="relu", layer_scale_init=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.mha = MultiHeadAttention()
        self.ffn = PositionWiseFNN()