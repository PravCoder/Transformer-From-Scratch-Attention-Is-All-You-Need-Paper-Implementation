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