from configs.model import TransformerArchitecture

# define a dict that maps the transformer-arch-type to the list strings of its corresponding special-tokens since different architectures have different special tokens
DEFAULT_SPECIAL_TOKENS: dict[
    TransformerArchitecture,
    list[str],
] = {
    TransformerArchitecture.ENCODER_ONLY: [
        "<PAD>",
        "<UNK>",
        "<CLS>",
        "<SEP>",
        "<MASK>",
    ],

    TransformerArchitecture.DECODER_ONLY: [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ],

    TransformerArchitecture.ENCODER_DECODER: [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ],
}