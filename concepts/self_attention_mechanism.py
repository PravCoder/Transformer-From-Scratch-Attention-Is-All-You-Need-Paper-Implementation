# FILE: implements self-attention mechanism or scaled dot product attention in numpy-only for the math, represents a single head.
import numpy as np
import math



"""
Methods:
    forward(): given batch of input sequence embeddings (one embedding per token, per row), it does scaled dot product attention 
               outputing weighted sum of values which is for each word its new presentation after looking at every other word including itself.
               for each token, a new representation (context vector) that encodes information from all tokens in the sequence, including itself.
Attributes:
    d_model: input embedding size
    d_k: dimensionality of key-vector for ith token, in the paper d_k = d_v = d_model / num-heads so = d_model
    d_v: dimensionality of value-vector for ith token 
    W_q: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into queries in xW_q = Q
    W_k: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into keys in xW_k =K
    W_v: ∈ ℝ^(d_model × d_k), learned projection weights matrix that maps input embeddings into values xW_v = V
"""
class SelfAttentionHead:

    def __init__(self, d_model, d_k=None, d_v=None):

        self.d_model = d_model
        self.d_k = d_model if d_k is None else d_k
        self.d_v = d_model if d_v is None else d_v

        # init projection matrices to small random values of correct shape, scaling to make sure variance stays reasonable
        self.W_q = np.random.randn(self.d_model, self.d_k) / np.sqrt(d_model)   
        self.W_k = np.random.randn(self.d_model, self.d_k) / np.sqrt(d_model)
        self.W_v = np.random.randn(self.d_model, self.d_k) / np.sqrt(d_model)

    # X: (batch, seq_len, d_model)
    def forward(self, X):
        
        # compute matrix of queries. Row Q_i is query vector for word i. ∈ ℝ^(n × d_k) with batch dim
        Q = np.dot(X, self.W_q)

        # compute matrix of keys. Row K_i is key vector for word i. ∈ ℝ^(n × d_k) with batch dim
        K = np.dot(X, self.W_k)

        # compute matrix of values. Row V_i is value vector for word i. ∈ ℝ^(n × d_k) with batch dim
        V = np.dot(X, self.W_v)

        print(f"{Q.shape=} matrix of queries")  # expected (1, 3, 4) 
        print(f"{K.shape=} matrix of keys")
        print(f"{V.shape=} matrix of values")

        # compute attention/similarity scores plus scaling, shape: (batch, seq_len, seq_len) = (batch, n, n)
        # right now K is (batch, seq_len, d_k). But we need it in shape (batch, d_k, seq_len) thats why we do the transpose so we can do:
        attention_scores =  np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.d_k)
        

        # add softmax along last axis (seq_len), every element is attention weight vector for ith word query, shape: (batch, seq_len, seq_len)
        weights_alpha = self.softmax(attention_scores)
        

        # weighted sum of values, each element is a new presentation of word i, it looks at entire sequence when we computed z_i for all words, that means each word looks at all words including itself. 
        # shape: (batch, seq_len, d_v)
        z = np.matmul(weights_alpha, V)

        print(f"{attention_scores.shape=}")
        print(f"{weights_alpha.shape=}")
        print(f"{z.shape=}")

        # each element is a vector that represents the token after attending to the entire sequence (including itself). It’s a weighted sum of all value vectors, where the weights come from the attention scores.
        # zi​ = summation from j=1 to n of (alpha_ij * V_j)
        return z

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)






def main():
    # create toy made up input for this numpy implementation
    X = np.array([
        [[1.0, 0.0, 1.0, 0.0],  # token 1
        [0.0, 2.0, 0.0, 2.0],   # token 2
        [1.0, 1.0, 1.0, 1.0]]   # token 3
    ])
    # shape: (1, 3, 4) = (batch, seq_len, d_model), seq_len = tokens in sequence, d_model = each token embedding size

    d_model=4
    head = SelfAttentionHead(d_model=d_model, d_k=d_model, d_v=d_model)   # set equalt to 4 because our embedding vector is dim 4


    print("\n==========INIT PROJECTION MATRICES==========")
    print(f"{X.shape=}")
    print(f"{d_model=}")
    print(f"{head.W_q.shape=} queries projection matrix")   # expected (4,4) = (d_model, d_k)
    print(f"{head.W_k.shape=} keys projection matrix")
    print(f"{head.W_v.shape=} values projection matrix")

    print("\n==========FORWARD PASS OF HEAD==========")
    head.forward(X=X)

if __name__ == "__main__":
    main()

"""
Validation since its random init:
- Rows of softmax weights ≈ 1.0
- Shapes match expectations
- Changing one token in X changes attention weights in a meaningful way
"""