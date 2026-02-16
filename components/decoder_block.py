import torch
import torch.nn as nn
import torch.nn.functional as F
from position_wise_fnn import PositionWiseFNN
from multi_head_attention_mechanism import MultiHeadAttention

"""
TODO:
- attention weights test: to confirm causal mask makes future attention 0 and pad mask amkes attention to PAD source positions 0
"""


"""
What: represents one decoder block or layer.
Methods:
    forward(): computes forward pass for a single decoder block which computes all 3 decoder sublayers
Attributes:
    d_model: dimension of each embedding vector for each token
    num_heads: number of attention heads.
    d_ff: hidden dimension for feed-forward network
    dropout: dropout probability used
    multi_head_attn_cls: reference to our multi-head-attention class, not an object it is the class itself

Note: this is post-norm to match original equations
"""
class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, multi_head_attn_cls):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.multi_head_attn_cls = multi_head_attn_cls
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # We define 2 different mha we need in decoder block.
        # Sublayer 1 [Masked-Multi-Head-Attention]:  an instance of multi-head-attn-class
        self.masked_multihead_attn = self.multi_head_attn_cls(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, heads=self.num_heads, dropout=self.dropout_p)
        # Sublayer 2 [Multi-Head Cross Attention Encoder-Decoder]:  an instance of multi-head-attn-class
        self.multi_head_cross_attn = self.multi_head_attn_cls(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, heads=self.num_heads, dropout=self.dropout_p)
        # Sublayer 3 [Position-Wise Feed Forward Network]: an instance of PositionWiseFNN class
        self.fnn = PositionWiseFNN(d_model=d_model ,d_ff=d_ff, dropout=self.dropout_p)

        # three separate LayerNorm modules, each normalizes over last dimension d_model
        # cannot use single LayerNorm each has its own learned parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # single dropout module reused in the block
        self.dropout = nn.Dropout(self.dropout_p)

    """ 
    What: computes forward pass of single decoder block
    Arguments:
        Y_lm1: (B, N_T, d_model). Y^(l-1) the input to current lth block, the target embeddings or output of previous block.
        H_N: (B, N_S, d_model). The final encoder output encoded source sequence contextualized representation of source sequences.
        M_causal: (B, 1, N_T, N_T) or braodcastable.
        M_pad: (B, 1, 1, N_S) or (B, 1, N_T, N_S)
    Returns:
        Y_l: final output of lth deocder block
    """
    def forward(self, Y_lm1, H_N, M_causal=None, M_pad=None):
        # ----- SUBLAYER-1 MASKED MULTI-HEAD SELF-ATTENTION  -----
        self_attn_out, _ = self.masked_multihead_attn(Q_in=Y_lm1, K_in=Y_lm1, V_in=Y_lm1, mask=M_causal)      # passing in previoous decoder block output, returns (out, alpha)
        Y_tilda_l = self.ln1(Y_lm1 + self.dropout(self_attn_out))

        # ----- SUBLAYER-2 MULTI-HEAD CROSS ATTENTION -----
        cross_attn_out, _ = self.multi_head_cross_attn(Q_in=Y_tilda_l, K_in=H_N, V_in=H_N, mask=M_pad)      # passing in previous sublayer output and encoder output
        Y_hat_l = self.ln2(Y_tilda_l + self.dropout(cross_attn_out))

        # ----- SUBLAYER-3 FNN -----
        fnn_out = self.fnn(Y_hat_l)
        Y_l = self.ln3(Y_hat_l + self.dropout(fnn_out))         # passing in previous sublayer output
        return Y_l
    
# helper functions to create test masks
def make_M_causal(B, N_T, device):
    m = torch.triu(torch.ones(N_T, N_T, dtype=torch.bool, device=device), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0).expand(B, 1, N_T, N_T)
def make_M_pad_from_src_ids(src_ids, pad_id):
    return (src_ids == pad_id).unsqueeze(1).unsqueeze(1)    # (B,1,1,N_S)

def tests():
    d_model, d_k, d_v, heads, d_ff = 16, 8, 8, 2, 32
    B, N_T, N_S = 2, 6, 7
    vocab_size, pad_id, dropout = 50, 0, 0.1
    device = "cpu"

    device = torch.device(device)

    # real embedding tables of size (vocab_size, d_model), creates a learnable matrix
    target_emb = nn.Embedding(vocab_size, d_model).to(device)
    source_emb = nn.Embedding(vocab_size, d_model).to(device)

    # fake token IDs, 
    target_ids = torch.randint(1, vocab_size, (B, N_T), device=device)  # target_ids[x] = entier targer seuqence for batch item x of shape (N_T), target_ids[x][y] = the token ID at position y in batch sequence x    , avoid pad by starting at 1
    source_ids = torch.randint(1, vocab_size, (B, N_S), device=device)  # random integers into tensor of shape (B, N_T/N_S)

    # set the last two tokens of the first source sequence to pad_id to test M_pad
    source_ids[0, -2:] = pad_id

    # real embeddings, inputs to decoder
    Y_lm1 = target_emb(target_ids)    # (B, N_T, d_model) -> plays role of Y^*l-1), takes each integer in target_ids and uses it as a row index to index target_emb and returns that row vector, just a lookup table.
    H_N = source_emb(source_ids)      # (B, N_S, d_model) -> plays role of H^(N)

    print(f"Y_lm1.shape: {Y_lm1.shape=}")

    # create masks
    M_causal = make_M_causal(B=B, N_T=N_T, device=device)
    M_pad = make_M_pad_from_src_ids(src_ids=source_ids, pad_id=pad_id)

    d_block = DecoderBlock(d_model=d_model, num_heads=heads, d_ff=d_ff, dropout=dropout, multi_head_attn_cls=MultiHeadAttention).to(device)
    d_block.train()
    out = d_block(Y_lm1=Y_lm1, H_N=H_N, M_causal=M_causal, M_pad=M_pad)     # forward pass, Y^(l)

    print("\n\n==========DECODER BLOCK MINIMAL TEST==========")
    print(f"Target IDs shape: {target_ids.shape=}")
    print(f"Y_lm1.shape: {Y_lm1.shape=}")
    print(f"M_causal.shape: {M_causal.shape=}")
    print(f"out.shape: {out.shape=}")

    assert out.shape == (B, N_T, d_model)
    print("[✅]PASS: decoder block shape test")
    return out


if __name__ == "__main__":
    tests()