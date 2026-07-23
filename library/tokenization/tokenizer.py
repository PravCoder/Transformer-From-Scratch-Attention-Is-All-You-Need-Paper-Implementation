"""
FILE: high-level interface that ties everything together for tokenizer: train(), encode(), decode(), save(), lead().
Just combines everything in vocabulary.py, bpe_trainer.py, bpe_encoder.py, bpe_decoder.py.
"""

from vocabulary import Vocabulary
from bpe_trainer import BPETrainer
from bpe_encoder import BPEEncoder
from bpe_decoder import BPEDecoder
# a typed variable to represent two paired tokens
TokenPair = tuple[str, str]


class BPETokenizer:


    def __init__(self) -> None:
        # represents the vocabulary of the tokenizer, which stores what tokens it knows
        self.vocabulary: Vocabulary | None = None
        # stores token-pair that should be mergerd which represents a merge-rrule
        self.merge_rules: list[TokenPair] = []

        # represents the tokenizer encoding algorithm
        self.encoder: BPEEncoder | None = None
        # represents the tokenizer decoding algorithm
        self.decoder: BPEDecoder | None = None

        # if the tokenizer has been trained or not
        self.is_trained = False

    def train(self, corpus: list[str], target_vocab_size: int, min_pair_freq: int = 1):
        # init the tokenizer's train
        trainer = BPETrainer(corpus=corpus, target_vocab_size=target_vocab_size, min_pair_freq=min_pair_freq)
        # train the tokenizer
        training_result = trainer.train()

        # save the trained vocabulary and merge rules
        self.vocabulary = training_result.vocabulary
        self.merge_rules = training_result.merge_rules.copy()

        # after training the tokenizer init its encoder and decoder
        self.encoder = BPEEncoder(vocabulary=self.vocabulary, merge_rules=self.merge_rules)
        self.decoder = BPEDecoder(vocabulary=self.vocabulary)

        self.is_trained = True

    """
    Convert raw text into its final BPE token representation
    """
    def tokenize(self, text: str) -> list[str]:
        return self.encoder.tokenize(text)

    """
    convert raw text into BPE token IDs
    """
    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    """
    Convert BPE token ids back into text
    """
    def decode(self, token_ids: list[int]) -> str:
        return self.decoder.decode(token_ids)

    
    # -- helper funcs ---
    def get_vocabulary(self) -> dict[str, int]:
        self.check_is_trained()
        assert self.vocabulary is not None
        return self.vocabulary.token_to_id

    def get_merge_rules(self) -> list[TokenPair]:
        self.check_is_trained()
        return self.merge_rules.copy()

    def get_vocab_size(self) -> int:
        self.check_is_trained()
        assert self.vocabulary is not None
        return len(self.vocabulary)

    def check_is_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "The tokenizer must be trained before calling "
                "tokenize(), encode(), or decode()."
            )


if __name__ == "__main__":

    print("------ Testing Complete BPE Tokenizer ------\n")

    corpus = [
        "hello",
        "help",
        "helmet",
    ]

    tokenizer = BPETokenizer()

    tokenizer.train(
        corpus=corpus,
        target_vocab_size=10,
    )

    original_text = "hello"

    tokens = tokenizer.tokenize(original_text)
    token_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(token_ids)

    print("Vocabulary:")
    print(tokenizer.get_vocabulary())

    print("\nMerge Rules:")
    print(tokenizer.get_merge_rules())

    print("\nOriginal Text:")
    print(original_text)

    print("\nBPE Tokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded Text:")
    print(decoded_text)

    print("\nRound-trip successful:")
    print(original_text == decoded_text)