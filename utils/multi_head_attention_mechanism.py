import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Methods:
    forward(): computes forward pass of single MHA head
Attributes:
    d_model: input embeeding dimension for a token
    d_k: dimension of keys, dim of key vector for ith word
    d_v: dimension of values, dim of value vector for ith word
    dropout: dropout rate for attention weights. Probability that nay given neurons output will be set ot zero.
    heads: number of heads for MHAM

    
"""
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads

        # create linear transformation layers for query, key, value weight matrices
        # big-projection-query-weight-matrix where each query-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * key_vec_dim)
        self.W_q = nn.Linear(d_model, heads * d_k)
        # big-projection-key-weight-matrix where each key-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * key_vec_dim)
        self.W_k = nn.Linear(d_model, heads * d_k)
        # big-projection-value-weight-matrix where each value-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * value_vec_dim)
        self.W_v = nn.Linear(d_model, heads * d_v)


def main():
    B, N, d_model, d_k, d_v, heads = 2, 5, 16, 8, 8, 2  # B=batch-size, N=seq-len, d_model=input-embed-dim, d_k=key-vec-dim, hea
    print(f"{B=}, {N=}, {d_model=}, {d_k=}, {d_v=}, {heads=}")

    print("==========CREATE MULTI-HEAD OBJECT==========")
    head = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_k, heads=heads)  
    print(f"{head.W_q=}")   # expected (16, (2*8)) = (16, 16s)
    print(f"{head.W_k=}")
    print(f"{head.W_v=}")

if __name__ == "__main__":
    main()