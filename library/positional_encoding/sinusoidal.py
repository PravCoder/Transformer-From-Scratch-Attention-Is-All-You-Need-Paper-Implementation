"""
FILE: Implements sinusoidal positional encodding from scratch.

Input:
    H: [B, L, d_model]

    B = batch size
    L = current sequence length
    d_model = dimension of each embedding/model vector

Output:
    Z: [B, L, d_model]

    Z[b][t] = H[b][t] + P[t]

Positional Encoding Matrix:
    P: [max_sequence_length, d_model]
    P[t] = positional encoding vector for sequence position t
    P[t][j] = positional encoding scalar for sequence position t and model dimension j
"""

import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_sequence_length: int):
        super().__init__()      # inherit for nn module so it makes it easy to integrate cleanly with the rest of the pytorch model
        # length of embedding vector
        self.d_model = d_model
        # the maximum number of tokens in each sequence
        self.max_sequence_length = max_sequence_length

        P = self.create_positional_matrix()

        # P is part of the module but it is NOT a learnable parameter, register buffer means P is stored with the model, Pmoves with the model between CPU/GPU
        self.register_buffer("P", P)

    """
    Creates the sinusoidal positional encoding matrix P. Doesn't take in input H because it is only created once, because it is used for every sequence in every batch.
    This is purposefully written in a slow non-vecotrized loop-version for learning & its fine to leave it slow because P matrix is only created once and used for every batch and every sequence.
    
    P shape:
        [max_sequence_length, d_model]

    P[pos]:
        positional encoding vector for sequence position pos
    """
    def create_positional_matrix(self) -> torch.Tensor:
        # create empty positionla encoding matrix with correct shape [L, d_model]
        P = torch.empty(self.max_sequence_length, self.d_model)

        # iterate all sequence-positions because we need to create a positional-embedding-vector for every position in the sequence
        for pos in range(self.max_sequence_length):
            # iterate all positions in the sequence, j is the second index for P
            for j in range(self.d_model):
                # if its an even position -> use sine equation
                if j % 2 == 0:
                    # j = 2i -> solve for i
                    i = j // 2 
                    # use sine formula
                    P[pos][j] = torch.sin(torch.tensor(pos / (10000 ** ((2 * i) / self.d_model))))
                # if its an odd position -> use cosine equation
                elif j % 2 != 0:
                    # j = 2i + 1 -> solve for i
                    i = (j - 1) // 2
                    # use cosine formula
                    P[pos][j] = torch.cos( torch.tensor( pos / ( 10000 ** ((2 * i) / self.d_model))))

        return P



    """
    Adds positional encoding vectors to embedding vectors.

    Input:
        H: [B, L, d_model]
        H[b] = all embedding vectors for sequence b
        H[b][t] = embedding vector for the token at sequence position t in sequence b

    Positional matrix:
        P: [max_sequence_length, d_model]
        P[t] = positional encoding vector for sequence position t

    Output:
        Z: [B, L, d_model]
        Z[b][t] = position-aware representation for the token at sequence position t in sequence b

    Equation:
        Z[b][t] = H[b][t] + P[t]
    """
    def add_positional_encoding(self, H: torch.Tensor) -> torch.Tensor:
        sequence_length = H.shape[1]
        # the sequence-length of output-tensor-H cannot exceed hte max-sequence-length
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds " f"maximum positional encoding length " f"{self.max_sequence_length}.")
        # the length of the embedding vector has to be d_model
        if H.shape[2] != self.d_model:
            raise ValueError( f"Expected embedding dimension {self.d_model}, " f"but received {H.shape[2]}." )
        # just in case the currrent-sequence-length for H's sequences is less than the max-sequence-length, we only select the positional-encoding-vectors up to it
        P_cur = self.P[:sequence_length]
        # for each sequence b in batch in H, for each embedding vector from H,  for the current sequence position t, add the corresponding sequence position t positional encoding vector from P.
        # for every position in every sequenece in H,  add its positional-encoding-vector to that position-embedding-vector
        Z = H + P_cur

        return Z


    """
    Forward pass of sinusoidal positional encoding. Needed because nn.Module
    Input:
        H: [B, L, d_model]
    Output:
        Z: [B, L, d_model]
    """
    def forward(self, H) -> torch.Tensor:
        Z = self.add_positional_encoding(H)
        return Z




# run: python -m positional_encoding.sinusoidal, library/
if __name__ == "__main__":
    print("========== TESTING SINUSOIDAL POSITIONAL ENCODING ==========")

    # Tiny fake model configuration
    batch_size = 2
    sequence_length = 6
    d_model = 4
    max_sequence_length = 6

    # Fake embedding tensor H
    #
    # Shape:
    #     [B, L, d_model]
    #
    # B = 2
    # L = 6
    # d_model = 4
    H = torch.tensor([
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70, 0.80],
            [0.90, 1.00, 1.10, 1.20],
            [1.30, 1.40, 1.50, 1.60],
            [1.70, 1.80, 1.90, 2.00],
            [2.10, 2.20, 2.30, 2.40]
        ],
        [
            [2.50, 2.60, 2.70, 2.80],
            [2.90, 3.00, 3.10, 3.20],
            [3.30, 3.40, 3.50, 3.60],
            [3.70, 3.80, 3.90, 4.00],
            [4.10, 4.20, 4.30, 4.40],
            [4.50, 4.60, 4.70, 4.80]
        ]
    ], dtype=torch.float32)


    # --------------------------------------------------
    # 1. INITIALIZE POSITIONAL ENCODING
    # --------------------------------------------------

    print("\n--- INITIALIZE POSITIONAL ENCODING ---")

    positional_encoding = SinusoidalPositionalEncoding(
        d_model=d_model,
        max_sequence_length=max_sequence_length
    )

    print("\nPositional Encoding Matrix P:")
    print(positional_encoding.P)

    print("\nP Shape:")
    print(positional_encoding.P.shape)

    assert positional_encoding.P.shape == (
        max_sequence_length,
        d_model
    )


    # --------------------------------------------------
    # 2. VERIFY FIRST POSITIONAL ENCODING VECTOR
    # --------------------------------------------------

    print("\n--- VERIFY P[0] ---")

    print("\nP[0]:")
    print(positional_encoding.P[0])

    # For position 0:
    #
    # sin(0) = 0
    # cos(0) = 1
    #
    # Therefore for d_model = 4:
    #
    # P[0] = [0, 1, 0, 1]

    expected_P0 = torch.tensor([
        0.0,
        1.0,
        0.0,
        1.0
    ])

    print("\nExpected P[0]:")
    print(expected_P0)

    assert torch.allclose(
        positional_encoding.P[0],
        expected_P0,
        atol=1e-6
    )


    # --------------------------------------------------
    # 3. VERIFY A SPECIFIC ELEMENT OF P
    # --------------------------------------------------

    print("\n--- VERIFY SPECIFIC P ELEMENT ---")

    # P[1][0]
    #
    # pos = 1
    # dimension = 0
    # dimension 0 is even
    # i = 0
    #
    # P[1][0] =
    # sin(1 / 10000^(0 / d_model))
    #
    # = sin(1)

    expected_value = torch.sin(torch.tensor(1.0))

    print("\nP[1][0]:")
    print(positional_encoding.P[1][0])

    print("\nExpected sin(1):")
    print(expected_value)

    assert torch.allclose(
        positional_encoding.P[1][0],
        expected_value,
        atol=1e-6
    )


    # --------------------------------------------------
    # 4. FORWARD PASS
    # --------------------------------------------------

    print("\n--- FORWARD PASS ---")

    Z = positional_encoding(H)

    print("\nInput Embedding Tensor H:")
    print(H)

    print("\nH Shape:")
    print(H.shape)

    print("\nOutput Tensor Z:")
    print(Z)

    print("\nZ Shape:")
    print(Z.shape)

    # Positional encoding does NOT change tensor shape.
    #
    # H:
    #     [B, L, d_model]
    #
    # Z:
    #     [B, L, d_model]

    assert Z.shape == H.shape

    assert Z.shape == (
        batch_size,
        sequence_length,
        d_model
    )


    # --------------------------------------------------
    # 5. VERIFY H + P
    # --------------------------------------------------

    print("\n--- VERIFY Z[b][t] = H[b][t] + P[t] ---")

    # Test:
    #
    # b = 0
    # t = 2
    #
    # Z[0][2] should equal:
    #
    # H[0][2] + P[2]

    expected_vector = (
        H[0][2]
        + positional_encoding.P[2]
    )

    print("\nH[0][2]:")
    print(H[0][2])

    print("\nP[2]:")
    print(positional_encoding.P[2])

    print("\nExpected H[0][2] + P[2]:")
    print(expected_vector)

    print("\nActual Z[0][2]:")
    print(Z[0][2])

    assert torch.allclose(
        Z[0][2],
        expected_vector,
        atol=1e-6
    )


    # --------------------------------------------------
    # 6. VERIFY SAME P[t] IS USED FOR EVERY SEQUENCE
    # --------------------------------------------------

    print("\n--- VERIFY SAME P[t] IS USED ACROSS BATCH ---")

    # Both sequences use P[2] at sequence position 2.
    #
    # Therefore:
    #
    # Z[0][2] - H[0][2] = P[2]
    #
    # Z[1][2] - H[1][2] = P[2]

    position_added_sequence_0 = Z[0][2] - H[0][2]
    position_added_sequence_1 = Z[1][2] - H[1][2]

    print("\nPosition vector added to sequence 0:")
    print(position_added_sequence_0)

    print("\nPosition vector added to sequence 1:")
    print(position_added_sequence_1)

    print("\nP[2]:")
    print(positional_encoding.P[2])

    assert torch.allclose(
        position_added_sequence_0,
        positional_encoding.P[2],
        atol=1e-6
    )

    assert torch.allclose(
        position_added_sequence_1,
        positional_encoding.P[2],
        atol=1e-6
    )


    # --------------------------------------------------
    # 7. VERIFY P IS NOT LEARNABLE
    # --------------------------------------------------

    print("\n--- VERIFY P IS NOT A LEARNABLE PARAMETER ---")

    parameter_names = [
        name
        for name, _ in positional_encoding.named_parameters()
    ]

    buffer_names = [
        name
        for name, _ in positional_encoding.named_buffers()
    ]

    print("\nLearnable parameters:")
    print(parameter_names)

    print("\nBuffers:")
    print(buffer_names)

    # P should NOT appear as a learnable parameter.
    assert "P" not in parameter_names

    # P SHOULD be registered as a buffer.
    assert "P" in buffer_names


    print(
        "\n========== ALL SINUSOIDAL POSITIONAL ENCODING TESTS PASSED =========="
    )