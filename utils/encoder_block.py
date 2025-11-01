import math
import torch
from torch import nn, Tensor
from multi_head_attention_mechanism import MultiHeadAttention
from position_wise_fnn import PositionWiseFNN


"""
What: represents an single encoder block not the Encoder itself. The encoder is the composition of N encoder blocks. That is why there is no l layer loop in the forward pass.
      So you loop over N identical encoder layers is the Encoder module not inside each block. Encoder layer and encdoer block are almost used interchangably
Methods:
    forward(): tbd
Attributes:
    d_model: input embedding size.
    num_heads: number of attention heads for MHA. 
    d_ff: inner dim of position-wise feed-forward network
    activation: act-function for FNN.

Notes: post norm.
"""
class EncoderBlock(nn.Module):
    

    def __init__(self, d_model, num_heads, d_ff, attn_dropout=0.0, residual_dropout_prob=0.0, activation="relu", layer_scale_init=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.residual_dropout_prob = residual_dropout_prob
        
        self.mha = MultiHeadAttention(d_model=d_model, d_k=d_model, d_v=d_model, heads=num_heads)
        self.ffn = PositionWiseFNN(d_model=d_model, d_ff=d_ff, activation="relu")

        # define the sub-layers for each N identical layers for one encoder block
        # create layer-normalization-mod for 1st noramlization in attention sublayer, that normalizes tehr euslt of the attention sublayer, and across last dimension the embedding dimension, to stabalize activations & gradeints
        self.ln1 = nn.LayerNorm(d_model)
        
        # create another layer-normalization-mod for 2nd normalization in feedforward sublayer, that noramlizes after the feedforward sublayer
        self.ln2 = nn.LayerNorm(d_model)

        # create two seperate dropout layers one for each sublayer of the encoder block
        # this randomly sets each element of the input tensor to zero with the probability of residual_dropout_prob, this is applied to the output of the MHA
        self.drop1 = nn.Dropout(residual_dropout_prob)
        self.drop2 = nn.Dropout(residual_dropout_prob)  # is applied to the output of the feedforwad network

    """
    What: computes forward pass of one encoder block, not Encoder. 
    Arguments:
        X_l: input to encdoer layer l with shape (B, N, d_model)
        return_attn: boolean if True it also returns average attention weights (B, N, N)
    Returns:
        Y_l: output Y^(l) with shape (B, N, d_model), the high-level contextual encdoed embeddings with semantic meaning, context, relationships between all elements in the input data.
    """
    def forward(self, X_prev, return_attn=False):
        # only do post-norm path meaning we apply the layer-norms after the multi-head and fnn sublayers
        # this is the token embeddings entering the current encoder block, i.e the output from the previous layer or the embbedings if this is the first layer
        X = X_prev

        # ---SUB-LAYER 1---:
        # compute forward pass of multi-head-attention passing X ∈ (B, N, d_model) as input, returns
        # attn_out: (B, N, d_model) which is new contextulized embeeddings, where attn_out[b, i, :] = new emebbeding for ith token in bth sequence
        # attn_weights: (B, H, N, N), where alpha[b, h, j, k] = how much token k's value contributes to token j's new representation in head h of batch b, optionally used for visualization
        attn_out, attn_weights = self.mha(X)
        # apply first dropout to attention-output, equation is Z^(l) randomly zeros out parts during training to regularize the model
        attn_out = self.drop1(attn_out) 
        # first add the original input to this layer (residial) to the attention output, this normalizes the features, this is equation 1.2 in notes, (B, N, d_model)
        y_l = self.ln1(X + attn_out)     # output of first sublayer of encoder layer

        # ---SUB-LAYER 2---:
        # apply forward-pass of position-wiase feedforward network to each token independlty, equation 2.1
        # 2-layer neural network ReLU(y^l*W1 + b1)W2 + b2, (B, N, d_model)
        ffn_out = self.ffn(y_l)
        # applies dropout to the FFN output, randomly zeros some elements to regularize second sub-layer
        ffn_out = self.drop2(ffn_out)   
        # first fnn-output back to its input the output of first sublayer for residual learning (residual connection), normalization
        out_x_l = self.ln2(y_l + ffn_out)   # equation 2.2

        return out_x_l, attn_weights if return_attn else (out_x_l, None)


def main():
    B, N, d_model = 2, 5, 32
    H = 4   # num of heads
    d_ff = 64   # dim of position-wise-fnn

    print("\n\n==========ENCODER BLOCK MINIMAL TEST==========")
    print(f"{B=}, {H=}, {d_ff=}")
    
    X = torch.randn(B, N, d_model)  # create a random tensor of input embeddings spahe (batch_size, seq-len, embedding-size)
    print(f"{X.shape=}")

    encoder_block = EncoderBlock(d_model=d_model, num_heads=H, d_ff=d_ff, attn_dropout=0.1, residual_dropout_prob=0.1, activation="gelu")

    y, attn = encoder_block(X, return_attn=True)

    print(f"\n{y.shape=}")  # expected [2, 5, 32] = (B, N, d_model)
    print(f"{attn.shape=}") # expected [2, 4, 5, 5] = (B, H, N, N)


if __name__ == "__main__":
    main()



