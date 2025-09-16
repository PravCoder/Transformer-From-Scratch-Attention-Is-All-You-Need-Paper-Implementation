import math
import torch
import torch.nn as nn


"""
Methods:
    forward():
Attributes:
    d_model: input embedding dimension
    d_k: dimension of keys, dim of key vector for ith word, we will use it for queries also
    d_v: dimension of values, dim of value vector for ith word
    dropout: dropout rate for attention weights. Probability that any given neurons output will be set to zero 
    W_q: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into queries in xW_q = Q
    W_k: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into keys in xW_k =K
    W_v: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into values xW_v = V

"""
class SelfAttentionHead(nn.Module):

    def __init__(self, d_model, d_k, d_v, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        # creates dropout-layer for layer neural network, given the dropout probability
        self.dropout = nn.Dropout(dropout)  

        # create linear transformation layer for queries projection weight matrix, shape (d_model, d_k) = (input_embedding_dim, key_vec_dim), no bias vector
        self.W_q = nn.Linear(d_model, d_k,  bias=False)
        # create linear transformation layer for keys projection weight matrix, shape (d_model, d_k) = (input_embedding_dim, key_vec_dim)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        #  # create linear transformation layer for values projection weight matrix, shape (d_model, d_k) = (input_embedding_dim, key_vec_dim)
        self.W_v = nn.Linear(d_model, d_k, bias=False)