import math
import torch
from torch import nn, Tensor

"""
Methods:
    forward(): tbd
Attributes:
    d_model: input embedding size.
    d_ff: inner dim of position-wise feed-forward network
Notes:
    B = batch size
    N = seqence size, number of tokens in sequence
"""
class PositionWiseFNN(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout=0.1
        self.activation = activation

        # define position-wise feedforward network that transforms each token embedding independently
        # sequential just wraps multiple layers into single thing that applies them in order of defined below
        self.net = nn.Sequential(
            # fully connected layer, input shape: (B, N , d_model), output shape: (B, N, d_ff), (inp, out) of layer is (d_mod, d_ff)
            # each token embedding of size d_model is projected into larger hidden dimension d_ff, this layer has d_ff number of nodes
            nn.Linear(d_model, d_ff),   
            # activation function between the two linear transformations, just applys it to the Z's of prev layer, and passes activations to next layer
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            # applies element-wise dropout after the activations, where indivudal nodes in layer are randomly set to zero
            nn.Dropout(dropout),
            # another fully connected layer which has d_model number of nodes, that projects back to the model dimension, input shape: (B, N, d_ff), output shape: (B, N, d_model)
            nn.Linear(d_ff, d_model),
            
        )
        # together its FNN(x) = W2(Dropout(Activation(W1(x))))

    # define the forward pass for the feedforward neural network
    # X: (B, N, d_model), is the encoders current embeddings of all tokens input.
    # where each vector X[b][i] = current token embedding vector for h^(l)_b_i sequence-b token-i
    def forward(self, X):
        # net() is the sequential mathematical equation for this network, everything is applied independently to each token position
        return self.net(X)


def main():
    # 2 sequences, 4 tokens each, 8-dim embeddings per token
    B, N, d_model = 2, 4, 8
    d_ff = 16

    print("\n=====POSITION-WISE FNN TESTS======")

    print(f"{B=}, {N=}, {d_model=}, {d_ff=}")

    torch.manual_seed(0)    # so results can reproduce, same rand nums each time program is ran

    X = torch.randn(B, N, d_model)  # create random input tensor of input embeddings shape

    fnn = PositionWiseFNN(d_model=d_model, d_ff=d_ff, activation="relu")    # create the network

    Y = fnn(X)      # compute forward pass

    print("\nInput shape: ", X.shape)
    print("Output shape: ", Y.shape)    # input and output shape should be the same
    assert X.shape == Y.shape, "X shape is different from Y shape"

    print("\nInput Sample Sequence 0th:")
    print(X[0])

    print("\nOutput Samples Sequence 0th:")
    print(Y[0])

    # for sanity check compute difference then mean to make sure they are not stricktly equal
    d = (Y - X).abs().mean().item()
    print(f"\nAverage absolute difference between input and output: {d} ")



if __name__ == "__main__":
    main()