"""
FILE: our own embedding layer implemented from scratch that performs the embedding look up instead of using pytorch's. 
Look at notion + hand notes
"""

import torch
from torch import nn

"""
Converts token IDs into embbeding vectors that are learnable.

Input:
    X: [B, L]
    B = batch size, L = sequence length
    note: has multiple sequences, each sequence is represents as a row of token IDs

Output Embedding Tensor:
    H: [B, L, d_model]
    note: as multiple sequences, each sequence is represented as multiple embedding vectors, one embedding vector per token in sequence

Embedding Matrix Learnable paramter:
    E: [V, d_model]

EmbeddingLayer inherits from torch neural network module because it is a learnable component. This means EmbeddingLayer is a type of Pytorch neural-network module.
Because it inherits from nn.Module torch automatically registers self.E as a model parameter. self. E should receive gradients during backpropagation during backprogation and be updated by the optimizer.
"""
class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.V = vocab_size         # number of tokens in vocab
        self.d_model = d_model      # dim of embedding vectors

        self.E = self.init_embedding_matrix()   # embedding matrix which is learnable parameter.

    """
    Creates the embedding matrix E of shape [V, d_model].
    Where:
        E[i] = embedding vector of token ID i which is size d_model
    """
    def init_embedding_matrix(self):
        # create the embedding matrix as a tensor with shape [V, d_model], make it a learnable model parameter.
        self.E = nn.Parameter(torch.empty(self.V, self.d_model))
        # insert inital random floating points into matrix E
        nn.init.normal_(self.E, mean = 0.0, std = 0.02)

        return self.E


    """
    Forward pass embedding layer performs embedding lookup.
    Input: 
        X of shape [B, L], where B = batch size, L = sequence length
        X[b] = sequence of token IDs for batch sequence b
        X[b][t] = the token ID at sequence position t for batch sequence b
    Output: 
        H of shape [B, L, d_model] the output embedding tensor
        H[b] = all token embedding vectors for sequence b in batch
        H[b][t] = the embedding vector for token t in batch sequence b.            
    """
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # compute the output-embedding-tensor of shape [B, L, d_model]
        # for every token ID in X, get the corresponding embedding-vector-row from embedding-matrix-E, store it in the output-embedding-vector
        H = self.E[X]

        return H




# run: python -m embeddings.embedding, library/
if __name__ == "__main__":
    print("========== TESTING EMBEDDING LAYER ==========")

    # Tiny fake model configuration
    vocab_size = 14
    d_model = 4

    # One fake batch:
    # B = 2
    # L = 6
    X = torch.tensor([
        [8, 2, 3, 13, 10, 10],
        [9, 13, 10, 10, 10, 10]
    ], dtype=torch.long)


    # --------------------------------------------------
    # 1. INITIALIZE EMBEDDING LAYER
    # --------------------------------------------------
    print("--- INITIALIZE EMBEDDING LAYER ---")
    embedding = EmbeddingLayer(
        vocab_size=vocab_size,
        d_model=d_model
    )

    print("\nEmbedding Matrix E:")
    print(embedding.E)

    print("\nEmbedding Matrix Shape:")
    print(embedding.E.shape)

    assert embedding.E.shape == (vocab_size, d_model)


    # --------------------------------------------------
    # 2. FORWARD PASS
    # --------------------------------------------------
    print("--- FORWARD PASS ---")
    H = embedding(X)

    print("\nInput Token IDs X:")
    print(X)

    print("\nInput Shape:")
    print(X.shape)

    print("\nOutput Embedding Tensor H:")
    print(H)

    print("\nOutput Shape:")
    print(H.shape)

    # X: [B, L] -> H: [B, L, d_model]
    assert H.shape == (2, 6, d_model)


    # --------------------------------------------------
    # 3. VERIFY EMBEDDING LOOKUP
    # --------------------------------------------------
    print("--- VERIFY EMBEDDING LOOKUP ---") 
    # X[0, 0] = 8
    # Therefore:
    # H[0, 0] must equal E[8]

    print("\nToken ID X[0, 0]:")
    print(X[0, 0])

    print("\nE[8]:")
    print(embedding.E[8])

    print("\nH[0, 0]:")
    print(H[0, 0])

    assert torch.equal(H[0, 0], embedding.E[8])


    # --------------------------------------------------
    # 4. VERIFY SAME TOKEN -> SAME EMBEDDING
    # --------------------------------------------------
    print("---  VERIFY SAME TOKEN -> SAME EMBEDDING ---") 
    # Token ID 10 appears multiple times.
    # Every occurrence should retrieve E[10].

    assert torch.equal(H[0, 4], embedding.E[10])
    assert torch.equal(H[0, 5], embedding.E[10])
    assert torch.equal(H[1, 2], embedding.E[10])

    assert torch.equal(
        H[0, 4],
        H[1, 2]
    )


    # --------------------------------------------------
    # 5. VERIFY E IS A LEARNABLE PARAMETER
    # --------------------------------------------------
    print("---  VERIFY E IS A LEARNABLE PARAMETER ---") 
    print("\nIs E a model parameter?")
    print(isinstance(embedding.E, torch.nn.Parameter))

    print("\nDoes E require gradients?")
    print(embedding.E.requires_grad)

    assert isinstance(embedding.E, torch.nn.Parameter)
    assert embedding.E.requires_grad


    print("\n========== ALL EMBEDDING TESTS PASSED ==========")