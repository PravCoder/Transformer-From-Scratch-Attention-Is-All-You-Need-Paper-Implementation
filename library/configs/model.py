
from enum import Enum

# just a class to define what type of transformer architecture we are building, we need to keep track of this for implementation details
class TransformerArchitecture(str, Enum):
    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
