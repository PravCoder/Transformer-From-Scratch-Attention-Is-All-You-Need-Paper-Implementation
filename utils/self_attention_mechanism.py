import math
import torch
import torch.nn as nn


"""
Methods:
    forward():
Attributes:
    d_model: input embedding dimension for a token
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
    
    """
    What: Forward pass for one self-attention head. Just implement the equations.
    Arguments:
        X: (batch, seq_len, d_model) input embeddings (number of sequences, length of sequence, embedding dimension for a token), N=seq_len
        mask: optional, TBD
    Returns:
        z: (batch, seq_len, d_k), the head-output-weighted-sum-of-values
        attention_weights: (batch, seq_len, seq_len), the attention weights after sclaing + softmax
    """
    def forward(self, X, mask=None):
        # calling forward pass of linear-layer-W_q passing in input X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k), where Qi is query-vector for ith token
        Q = self.W_q(X) 
        # calling forward pass of linear-layer-W_k passing in input X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k), where Ki is key-vector for ith token
        K = self.W_k(X)
        # calling forward pass of linear-layer-W_v passing in put X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k) where Vi is value-vector for ith token
        V = self.W_v(X)

        print(f"Q: {Q.shape}")
        print(f"K: {K.shape}")
        print(f"V: {V.shape}")


def main():
    B, N, d_model, d_k = 2, 5, 16, 8    # B=batch-size, N=seq-len, d_model=input-embed-dim, d_k=key-vec-dim
    print(f"{B=}, {N=}, {d_model=}, {d_k=}")

    # create input embeddings (batch, seq_len, d_model) 
    X = torch.randn(B, N, d_model)
    # create self-attention-head
    head = SelfAttentionHead(d_model=d_model, d_k=d_k, d_v=d_k, dropout=0.1)  

    print("==========FORWARD PASS OF HEAD==========")
    
    # this does forward-pass because we inheriting from torch
    # expected: (2, 5, 8) for Q, K, V shapes      
    head(X)         
    

    

if __name__ == "__main__":
    main()