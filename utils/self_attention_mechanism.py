import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Methods:
    forward(): computes forward pass of single head
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
        # compute projection representations
        # calling forward pass of linear-layer-W_q passing in input X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k), where Qi is query-vector for ith token
        Q = self.W_q(X) 
        # calling forward pass of linear-layer-W_k passing in input X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k), where Ki is key-vector for ith token
        K = self.W_k(X)
        # calling forward pass of linear-layer-W_v passing in put X, (B, N, d_model) * (d_model, d_k) = (B, N, d_k) where Vi is value-vector for ith token
        V = self.W_v(X)

        print(f"Q: {Q.shape}")
        print(f"K: {K.shape}")
        print(f"V: {V.shape}")

        # scaled dot product attention scores: QK^T / √d_k
        # K.transpose(-2, -1) swaps the last two dimensions from (B, N, d_k) -> (B, d_k, N)
        scaled_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
        # apply mask
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(~mask, float("-inf"))

        # apply softmax to scaled-scores, shape (B, N, N)
        attention_weights_alpha = F.softmax(scaled_scores, dim=-1)
        # passing atten-scores as input into dropout layer
        attention_weights_alpha = self.dropout(attention_weights_alpha)

        # compute weighted sum of values (head output), z = alpha * V, shape (B, N, d_v/d_k)
        z = torch.matmul(attention_weights_alpha, V)


        print(f"\nz: {z.shape}")                                                
        print(f"attention_weights_alpha: {attention_weights_alpha.shape}")

        return z, attention_weights_alpha        # return head output & attention weights
        

def test_with_real_embeddings(B=2, N=5, d_model=16, d_k=8, d_v=8, dropout=0.1, vocab_size=50):
    print(f"{B=}, {N=}, {d_model=}, {d_k=}, {vocab_size=}")

    # creates an embedding look up table of shape (vocab_size, d_model), each row corresponds to the vector representation of one token-ID
    emb = nn.Embedding(vocab_size, d_model)
    # generates a batch of random token IDs of shape (B, N) = (batch_size, seq_len) = (number_of_sequences, seq_len).
    # each sequence has 5 token ids cause N=5=seq_len
    # each entry is an integer between and vocabsize-1
    # x_ids[0] -> first sequence [17, 42, 3, 11, 29]
    # x_ids[1] -> second sequence [8, 0, 14, 21, 7]
    # x_ids[1, 4] -> 5th token-id in 2nd sequence = 7, equal to single token-id-integer
    x_ids = torch.randint(0, vocab_size, (B,N))

    # uses embedding layer as a lookup
    # each token-id-integer in x_ids is replaced by its corresponding row from the embedding matrix
    # shape: (B, N, d_model)
    # x[0][5] = get the first sequence 5th token embedding vector of size d_model
    x_real = emb(x_ids)
    print(f"x_real: {x_real.shape}")


    head = SelfAttentionHead(d_model=d_model, d_k=d_k, d_v=d_k, dropout=0.1)  
    # expected shape z: (2, 5, 8)
    # expected shape attention_weights: (2, 5, 5)
    z, attention_weights = head(x_real)

    
    return z, attention_weights, head, emb
    

def check_backprop(head, emb, z):   # sanity gradient checks, attach simple dummy loss to confirm gradients flow
    target = torch.randn_like(z)    # dummy target same shape as z

    criterion = nn.MSELoss()
    loss = criterion(z, target)

    loss.backward()

    print("Grad for embedding table:", emb.weight.grad is not None)
    print("Grad for W_q:", head.W_q.weight.grad is not None)
    print("Grad for W_k:", head.W_k.weight.grad is not None)
    print("Grad for W_v:", head.W_v.weight.grad is not None)

    print(head.W_q.weight.grad.norm())  # to check non zero
    print(head.W_k.weight.grad.norm())
    print(head.W_v.weight.grad.norm())

def main():
    B, N, d_model, d_k = 2, 5, 16, 8    # B=batch-size, N=seq-len, d_model=input-embed-dim, d_k=key-vec-dim
    print(f"{B=}, {N=}, {d_model=}, {d_k=}")

    # create input embeddings (batch, seq_len, d_model), sample random
    X = torch.randn(B, N, d_model)
    print(f"X: {X.shape}")
    # create self-attention-head
    head = SelfAttentionHead(d_model=d_model, d_k=d_k, d_v=d_k, dropout=0.1)  

    print("==========FORWARD PASS OF HEAD - SAMPLE EMBEDDINGS==========")
    print(f"X: {X.shape}")
    # this does forward-pass because we inheriting from torch
    # expected: (2, 5, 8) for Q, K, V shapes      
    head(X)         


    print("\n==========REAL EMBEDDINGS: SINGLE-SELF-ATTENTION-HEAD FORWARD PASS==========")
    # expected shape z: (2, 5, 8)
    # expected shape attention_weights: (2, 5, 5)
    z, attention_weights, head, emb = test_with_real_embeddings(B=2, N=5, d_model=16, d_k=8, d_v=8, dropout=0.1, vocab_size=50)

    print("\n==========CHECK BACKPROP OF HEAD==========")
    check_backprop(head, emb, z)
    

if __name__ == "__main__":
    main()