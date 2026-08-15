"""
FILE: defines the raw-text training-example output produced by step 4.

For an encoder-decoder-transformer one raw training example contains a source text sequence and a target text sequence
""" 
from dataclasses import dataclass


"""
Each encoder-decoder models training example has a source-text & target-text. This is for step 4.
"""
@dataclass(frozen=True, slots=True)
class EncoderDecoderTextTrainingExample:

    source_text: str
    target_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_text, str):
            raise TypeError("source_text must be a string.")
        if not isinstance(self.target_text, str):
            raise TypeError("target_text must be a string.")
        if not self.source_text.strip():
            raise ValueError("source_text cannot be empty.")
        if not self.target_text.strip():
            raise ValueError("target_text cannot be empty.")

    def __repr__(self) -> str:      # for printing debug statementss
        return (
            "EncoderDecoderTextTrainingExample(\n"
            f"  source_text={self.source_text!r},\n"
            f"  target_text={self.target_text!r}\n"
            ")"
        )

"""
Represents a tokenized training example for encoder-decoder model. 
Has a source-sequence of token ids & target-sequence of token ids. (the tokenizer is used to encode EncoderDecoderTextTrainingExample). Step 5.
"""
@dataclass
class EncoderDecoderTokenizedTrainingExample:
    source_token_ids: list[int]
    target_token_ids: list[int]

    def __repr__(self) -> str:      # for printing debug statements
        return (
            "EncoderDecoderTokenizedTrainingExample(\n"
            f"  source_token_ids={self.source_token_ids},\n"
            f"  target_token_ids={self.target_token_ids}\n"
            ")"
        )

"""
Represents one encoder-decoder training example format expected by the model created at step 6.
"""
@dataclass
class EncoderDecoderModelTrainingExample:

    encoder_input_ids: list[int]            # token-ids fed into the encoder

    decoder_input_ids: list[int]            # token-ids fed into the decoder

    decoder_target_ids: list[int]           # the correct token IDs the decoder should predict


    def __repr__(self) -> str:      # for printing debug statements            
        return (
            "EncoderDecoderModelTrainingExample(\n"
            f"  encoder_input_ids={self.encoder_input_ids},\n"
            f"  decoder_input_ids={self.decoder_input_ids},\n"
            f"  decoder_target_ids={self.decoder_target_ids}\n"
            ")"
        )