# FILE: implements self-attention mechanism or scaled dot product attention in numpy-only for the math, represents a single head.
import numpy as np

"""
Methods:
    forward(): 
Attributes:
    d_model: input embedding size
    d_k: dimensionality of key-vector for ith token, in the paper d_k = d_v = d_model / num-heads so = d_model
    d_v: dimensionality of value-vector for ith token 
    W_q: ∈ ℝ^(d_model × d_k), learned projection matrix that maps input embeddings into queries
    W_k: ∈ ℝ^(d_model × d_k), learned projection matrix that maps input embeddings into keys
    W_v: ∈ ℝ^(d_model × d_k), learned projection matrix that maps input embeddings into values
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




def main():
    # create toy made up input for this numpy implementation
    X = np.array([
        [[1.0, 0.0, 1.0, 0.0],  # token 1
        [0.0, 2.0, 0.0, 2.0],   # token 2
        [1.0, 1.0, 1.0, 1.0]]   # token 3
    ])
    # shape: (1, 3, 4) = (batch, seq_len, d_model), seq_len = tokens in sequence, d_model = each token embedding size

    d_model=4
    head = SelfAttentionHead(d_model=d_model, d_k=d_model, d_v=d_model)   # set equalt o 4 because our embedding vector is dim 4


    print("\n==========INIT PROJECTION MATRICES==========")
    print(f"{d_model=}")
    print(f"{head.W_q.shape=} queries projection matrix")   # expected (4,4) = (d_model, d_k)
    print(f"{head.W_k.shape=} keys projection matrix")
    print(f"{head.W_v.shape=} values projection matrix")

if __name__ == "__main__":
    main()

"""
Validation since its random init:
- Rows of softmax weights ≈ 1.0
- Shapes match expectations
- Changing one token in X changes attention weights in a meaningful way
"""