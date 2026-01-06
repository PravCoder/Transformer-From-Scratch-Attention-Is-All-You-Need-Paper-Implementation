import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
What: represents multi head self attention mechanism
Methods:
forward(): computes forward pass of single MHA head
Attributes:
    d_model: input embedding dimension for a token
    d_k: dimension of keys, dim of key vector for ith word
    d_v: dimension of values, dim of value vector for ith word
    dropout: dropout rate for attention weights. Probability that nay given neurons output will be set ot zero.
    heads: number of attention heads for MHAM

"""
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads

        # create linear transformation layers for query, key, value weight matrices which store those weight matrices for each head i.
        # linear layers to project X -> Q, K, V for all heads at once
        # big-projection-query-weight-matrix where each query-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * key_vec_dim)
        self.W_q = nn.Linear(d_model, heads * d_k)
        # big-projection-key-weight-matrix where each key-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * key_vec_dim)
        self.W_k = nn.Linear(d_model, heads * d_k)
        # big-projection-value-weight-matrix where each value-weight-matrix for head i is stacked side by side, shape (input-embed-dim, num-heads * value_vec_dim)
        self.W_v = nn.Linear(d_model, heads * d_v)

        # final output projection
        self.W_o = nn.Linear(heads * d_v, d_model)

        # creates a dropout-layer that will only be applied to the attnetion weights or softmaz output, dropouts is applied in training and not in inference, torch handles automaticallys
        # prevents heads from focusing too much on tokens, reduces overfitting.
        self.attn_dropout = nn.Dropout(dropout) 

    """
    What: Forward pass for one multi-head attention. Just implement the equations
    Arguments:
        X: (batch, seq_len, d_model), input embeddings (number of sequences, length of sequence, embedding dimension for a token), N=seq_len.
    Returns:
        out: (B, N, d_model), new contextualized representation for each token. The output embeddings after MHA. Meaning for each token out[b, i, :] is the contextualized embedding  of token i in sequence b.
        alpha: (B, H, N, N), attention weights for each head.
    """
    def forward(self, X, print_info=False):
        B, N, _ = X.shape

        # project and split into heads
        # for queries, keys, values multiply the corresponding learned-project-weight-matrices with input-embedding-X
        # shape (batch size, seq-len, num of heads, vector size of keys and values)
        Q = self.W_q(X).view(B, N, self.heads, self.d_k)
        K = self.W_k(X).view(B, N, self.heads, self.d_k)
        V = self.W_v(X).view(B, N, self.heads, self.d_v)

        # reshape heads into (B, H, B, d_k/d_v), so each head gets is own slice
        Q = Q.transpose(1, 2)   # Q[b, i, j, :] = query vector for jth token in the ith head, from bth sequence in batch. What does query vector represent?
        K = K.transpose(1, 2)   # K[b, i, j, :] = key vector for jth token in the ith head, from bth sequence in batch.  What does key vector represent?
        V = V.transpose(1, 2)   # V[b, i, j, :] = value vector for jth token in the ith head, from bth sequence in batch.  What does value vector represent?

        # equation: Qi*K_i^T / root(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1) / (self.d_k **0.5))
        
        # compute attention weights alpha
        # alpha[b, i, j, k] = how much token k's value contributes to token j's new representation in head i of batch b         -- IMPORTANT
        alpha = torch.softmax(scores, dim=-1)   # shape: (B, H, N, N)
        alpha = self.attn_dropout(alpha)        # apply dropout to thse attention weights meaning some attention links between tokens are randomly removed 

        # compute weight sum of values
        # Z[i] = alpha_i * V_i this gives output of each head
        # Z[b, i, j, :] = summation_k=1_to_N (alpha[n, i, j, k] * V[b, i, k, :]), this is the actual new representation of token j, k-indexes all tokens in the sequence, each token k has a value vector V[b, i, k, :]
        Z = torch.matmul(alpha, V)  

        # concatenate heads, Z = concat(Z1,...,Zh), shape (B, N, (h * dv))
        Z = Z.transpose(1, 2).contiguous().view(B, N, self.heads * self.d_v)

        # compute final output projection, Z*W_o
        # Z*W_o[b, n, :] = gets the new repsentation-vector of bth sequence of nth token
        out = self.W_o(Z)

        
        if print_info == True:
            print(f"{out.shape=}")
            print(f"{alpha.shape=}")

        return out, alpha

def test_with_real_embeddings(B=10, N=20, d_model=16, d_k=8, d_v=8, heads=3, dropout=0.1, vocab_size=50):
    print(f"{B=}, {N=}, {d_model=}, {d_k=}, {d_v=} {vocab_size=}")

    # creates an embedding look up table of shape (vocab_size, d_model), each row corresponds to the vector representation of one token-ID
    emb = nn.Embedding(vocab_size, d_model)

    # generates a batch of random token IDs of shape (B, N) = (number of sequnces, seq_len)
    # each sequence has 20 token ids cause N=20=seq_len
    # each entry is an integer between 0 and vocab_size-1
    # x_ids[0] -> first sequence [24, 5, 6,..., 8, 17], where each number is a token-ID representing a token in this sequence
    # x_ids[1] -> second sequence [4, 9, 12,...., 15, 6]
    # x_ids[1, 2] -> 3rd token-id in 2nd sequence = 12, equal to single token-id-integer
    x_ids = torch.randint(0, vocab_size, (B, N))

    # uses embedding layer as lookup
    # each token-ID-int in x-ids is replaced by its corresponding row from the embedding matrix
    # shape: (B, N, d_model)
    # x_real[0][5] = get the first seqence 5th token embedding vector of size d_model
    x_real = emb(x_ids)
    print(f"x_real: {x_real.shape}")

    mha = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, heads=heads)
    out, attention_weights = mha(x_real)

    return out, attention_weights, mha, emb

def main():
    B, N, d_model, d_k, d_v, heads = 2, 5, 16, 8, 8, 2  # B=batch-size, N=seq-len, d_model=input-embed-dim, d_k=key-vec-dim, hea
    print(f"\n{B=}, {N=}, {d_model=}, {d_k=}, {d_v=}, {heads=}")
    X = torch.randn(B, N, d_model)

    print("==========CREATE MULTI-HEAD OBJECT==========")
    mha = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_k, heads=heads)  
    print(f"{mha.W_q=}")   # expected (16, (2*8)) = (16, 16)
    print(f"{mha.W_k=}")
    print(f"{mha.W_v=}")
    print(f"{X.shape=}")

    print("==========MULTI-HEAD FORWARD PASS==========")
    # expected out: (2, 5, 16)
    # expected alpha: (2, 2, 5, 5)
    out, alpha = mha(X, print_info=True)  

    print("\n==========REAL EMBEDDINGS: MULTI-ATTENTION-HEAD FORWARD PASS ==========")
    out, alpha, mha, emb = test_with_real_embeddings(B=10, N=20, d_model=16, d_k=8, d_v=8, heads=3, dropout=0.1, vocab_size=50)
    print(f"{out.shape=}")   # expected: (10, 20, 16)
    print(f"{alpha.shape=}")  # expected: (10, 3, 20, 20)

if __name__ == "__main__":
    main()