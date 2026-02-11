import torch
import torch.nn as nn
import torch.nn.functional as F
from position_wise_fnn import PositionWiseFNN


"""
What: represents one decoder block or layer.
Methods:
    forward(): computes forward pass for a single decoder block which computes all 3 decoder sublayers
Attributes:
    d_model: dimension of each embedding vector for each token
    num_heads: number of attention heads.
    d_ff: hidden dimension for feed-forward network
    dropout: dropout probability used
    multi_head_attn_cls: reference to our multi-head-attention class, not an object it is the class itself

Note: this is post-norm to match original equations
"""
class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, multi_head_attn_cls):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.multi_head_attn_cls = multi_head_attn_cls

        # Sublayer 1 [Masked-Multi-Head-Attention]:  an instance of multi-head-attn-class
        self.masked_multihead_attn = self.multi_head_attn_cls(d_model=self.d_model, heads=self.num_heads, dropout=self.dropout_p)
        # Sublayer 2 [Multi-Head Cross Attention Encoder-Decoder]:  an instance of multi-head-attn-class
        self.multi_head_cross_attn = self.multi_head_attn_cls(d_model=self.d_model, heads=self.num_heads, dropout=self.dropout_p)
        # Sublayer 3 [Position-Wise Feed Forward Network]: an instance of PositionWiseFNN class
        self.fnn = PositionWiseFNN(d_model=d_model ,d_ff=d_ff, dropout=self.dropout_p)

        # three separate LayerNorm modules, each normalizes over last dimension d_model
        # cannot use single LayerNorm each has its own learned parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # single dropout module reused in the block
        self.dropout = nn.Dropout(self.dropout)

    """
    
    """
    def forward(self, Y_lm1, H_N, M_casual, M_pad):
        # ----- SUBLAYER-1 ------
        self_attn_out = self.masked_multihead_attn(Q_in=Y_lm1, K_in=Y_lm1, V_in=Y_lm1)
