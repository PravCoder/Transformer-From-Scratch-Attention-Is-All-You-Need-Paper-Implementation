"""
FILE: implements the BPE Decoding algorithm
"""

from vocabulary import Vocabulary

class BPEDecoder:

    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    """
    Converts a list token-IDs into text using the tokenizer's trained vocabulary
    """
    def decode(self, token_ids: list[int]) -> str:
        # convert each token-strings to token-IDs
        tokens = self.convert_ids_to_tokens(token_ids)
        # combine all token-strings into one text
        text = self.concatenate_tokens(tokens)

        return text
    
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        # contains the tokens-strings
        tokens: list[str] = []

        # iterate every token-id in sequence and get its corresponding token-string using vocab and add it to our tokens
        for cur_token_id in token_ids:
            cur_token = self.vocabulary.get_token(cur_token_id)
            tokens.append(cur_token)

        return tokens

    def concatenate_tokens(self, tokens: list[str]) -> str:
        # given token-strings, join them to reconstruct the original text, ["hell", "o"] -> "hello"
        return "".join(tokens)


if __name__ == "__main__":
    from bpe_trainer import BPETrainer
    from bpe_encoder import BPEEncoder

    print("------ Testing BPE Decoder ------\n")

    corpus = [
        "hello",
        "help",
        "helmet",
    ]

    trainer = BPETrainer(
        corpus=corpus,
        target_vocab_size=10,
    )

    training_result = trainer.train()

    encoder = BPEEncoder(
        vocabulary=training_result.vocabulary,
        merge_rules=training_result.merge_rules,
    )

    decoder = BPEDecoder(
        vocabulary=training_result.vocabulary,
    )

    original_text = "hello"

    token_ids = encoder.encode(original_text)
    decoded_text = decoder.decode(token_ids)

    print("Original Text:")
    print(original_text)

    print("\nEncoded Token IDs:")
    print(token_ids)

    print("\nDecoded Text:")
    print(decoded_text)