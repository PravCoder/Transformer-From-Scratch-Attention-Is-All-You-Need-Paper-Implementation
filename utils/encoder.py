import torch
import torch.nn as nn
from encoder_block import EncoderBlock

# TBD: start from text -> do processing of transformer -> etc


"""
What: represents the entire encoder module not just one block.
Methods:
    forward(): single forward pass of an entire encoder through its N identical layers
Attributes:
    num_layers: the numebr of N identical layers/block stacke don top of each other in the fullencdoer. Controls the number of times token embeddings are refined through attention + feedforward sublayers
    d_model: the dimension of the embedding vector for each token. Controls the size fo all projection matrices in attention and feedforward networks.
    num_heads: number of attention heads used in multi-head attention mechanism
    d_ff: inner dimension of the feedforward network inside each encoder block
    dropout: probability of randomly zeroing elements during training, used in self.drop1 & self.drop2.
"""
class Encoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.layers = nn.ModuleList()       # create a list of encoder-block-modules, 

        # iterate number of N-identical layers
        # create a encoder-block representing each layer, passing in same parameters
        for _ in range(num_layers): 
            layer = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, residual_dropout_prob=dropout, activation=activation)
            self.layers.append(layer)

        # final layer normalization
        self.norm = nn.LayerNorm(d_model)

    """
    What: computes forward pass of an entire encoder by doing a composition of its encoder-block-N-identical-layers, starting from first encoder-block to the last encoder-block
    Arguments:
        X: the input embeddings (B, N, d_model), where X[b][n] = input-embedding-vector nth token in the  bth sequence in batch.
    Returns:
        output: final encoder ouptut (B, N, d_model), that is the high-level contextual representation of the input
        all_atnn: optional, list of attention tensors [(B, H, N, N)] *num_layers
    """
    def forward(self, X, return_all_attn=False):
        attn_list = [] if return_all_attn else None # attn_list=[α(1),α(2),…,α(N)]

        # iterate over all N-identical-encoder-layer-blocks
        for layer in self.layers:
            # compute forward pass of layer-l passing in current-embeddings-X through encoder-block-layer-l
            # if its first layer then its the X, else its the previous-layers output X_l-1. Each block computes the 2 sublayers:
            # sublayer-1: Z^(l) = MultiHead(X^(l-1)), Y^(l) = LayerNorm(X^(l-1) + Dropout(Z^(l)))
            # sublayer-2: F^(l) = FNN(Y^(l)), X^(l) = LayerNorm(Y^(l) + Dropout(F^(l)))
            X, attn = layer(X_prev=X, return_attn=return_all_attn)
            # now this is the input to the next encoder-layer l+1, across layers the embeddings are being contextualized deeper and deeper

            if return_all_attn:     # just store the attention matrix (B, H, N, N) for this layer in a list
                attn_list.append(attn)


        if return_all_attn:     # return with attention-list if they want
            return X, attn_list
        
        # final layer normalization for stability
        # after the last encdoer-layer-N, X holds the final encoded representation of our input sequence. 
        # X = H^(N) = f^N_enc(f^N-1_enc(...f^1_enc(X^0)...)) ∈ (B, N, d_model) 
        # X[b, n, :] = final contextualized embedding vector of bth token in bth sequence of size d_model, the vector stores the original meaning of the token from emeddings plus all the contextual information gather through N layer of multi-head attention and feedforward transformations
        # so every tokens embedding now knows about every other token in its sequence. 
        X = self.norm(X)        

        return X
    


def main():
    print("\n---------- Setup ----------")
    B, N = 2, 5         # batch size, sequence length
    vocab_size = 50     # numer of unqiue tokens
    d_model = 32        # embedding dimension per token
    num_heads = 4       # number attention heads for multi-head-attention
    d_ff = 64           # feedforward network inner dimension
    num_layer = 3       # number of N-identical encoder-layer-blocks
    num_layers = 3      # number of encoder blocks
    
    print(f"{B=}, {N=}, {vocab_size=}, {d_model=}, {num_heads=}, {d_ff=}, {num_layer=}")

    # creates a learned embedding layer, a lookup table that maps each token-ID to adense vector of continous values its embedding
    # when you tokenize a sentence oyu get the integer IDs for each token: x_ids = [12, 57, 9, 4, 88]
    # so the embedding layer converts each integer ID into a learned vector of d_model size
    # so this layer interally has a weight matrix (vocab_size, d_model), The entries of embedding.weight are learnable parameters. They get updated by backpropagation — just like the weights of any neural network layer.
    embedding = nn.Embedding(vocab_size, d_model)

    # create encoder-module defining hyperparams
    encdoer = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1, activation="relu")

    print("\n---------- Input ----------")
    x_ids = torch.randint(0, vocab_size, (B, N))    # create random integer token-IDS, (like frmo tokenized text) 
    # convert to embeddings (B, N, d_model)
    x_embed = embedding(x_ids)

    print(f"Input IDs shape: {x_ids.shape}")
    print(f"Input embeddings shape: {x_embed.shape}")

    print("\n---------- Forward pass ----------")
    out, attn_list = encdoer(x_embed, return_all_attn=True)

    print(f"Encoder ouptut shape: {out.shape}")
    print(f"Number of attention layers: {len(attn_list)}")
    print(f"Attention shape of layer 0: {attn_list[0].shape}")


if __name__ == "__main__":
    main()