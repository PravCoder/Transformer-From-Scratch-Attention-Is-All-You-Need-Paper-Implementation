"""
FILE: high-level interface that ties everything together for tokenizer: train(), encode(), decode(), save(), lead().
Just combines everything in vocabulary.py, bpe_trainer.py, bpe_encoder.py, bpe_decoder.py.
"""

from tokenization.vocabulary import Vocabulary
from tokenization.bpe_trainer import BPETrainer
from tokenization.bpe_encoder import BPEEncoder
from tokenization.bpe_decoder import BPEDecoder
from tokenization.special_tokens import TransformerArchitecture, DEFAULT_SPECIAL_TOKENS
from dataset.training_example import EncoderDecoderTextTrainingExample, EncoderDecoderTokenizedTrainingExample
# a typed variable to represent two paired tokens
TokenPair = tuple[str, str]


class BPETokenizer:

    # the tokenizer needs what transformer-architecture we are dealing with so it can add the special tokens, and you can also add extra tokens in addition to the default special tokens
    def __init__(self, architecture: TransformerArchitecture, additional_special_tokens: list[str] | None = None) -> None:
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

        # the transformer-architecture of this model
        self.architecture = architecture
        # extra special tokens different from the default special tokens
        self.additional_special_tokens = (
            additional_special_tokens.copy()
            if additional_special_tokens is not None
            else []
        )
        # store the special-token-string to token-id, even tho they are in the vocabulary, just for the getters
        self.special_token_ids: dict[str, int] = {}

    def train(self, corpus: list[str], target_vocab_size: int, min_pair_freq: int = 1):
        # init the tokenizer's train
        trainer = BPETrainer(corpus=corpus, target_vocab_size=target_vocab_size, min_pair_freq=min_pair_freq)
        # train the tokenizer
        training_result = trainer.train()

        # save the trained vocabulary and merge rules
        self.vocabulary = training_result.vocabulary
        self.merge_rules = training_result.merge_rules.copy()

        # add all the special tokens to vocabulary
        self.add_architecture_special_tokens()

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

    """
    Given the text training examples that was created from the corpus by the given training objective, use the 
    trained tokenizers encoder function to encode all the text training examples into token IDs. Returns list of tokenized training examples.
    """
    def encode_training_examples(self, training_examples: list[EncoderDecoderTextTrainingExample]) -> list[EncoderDecoderTokenizedTrainingExample]:
        # in order ot encode training examples the tokenizer must be trained, raise error if not
        self.check_is_trained()

        # stores the tokenized training examples objs
        tokenized_training_examples = []

        # iterate all text-training-examples
        for cur_training_example in training_examples:
            # encode the cur-text-training-example source-text to get source-sequence-token-ids
            source_token_ids = self.encode(cur_training_example.source_text)

            # encode the cur-text-training-example target-text to get the target-sequence-token-ids
            target_token_ids = self.encode(cur_training_example.target_text)

            # create the tokenized-training-example-obj
            cur_tokenized_training_example = EncoderDecoderTokenizedTrainingExample(source_token_ids=source_token_ids, target_token_ids=target_token_ids)
            # add to all tokenized-training-examples
            tokenized_training_examples.append(cur_tokenized_training_example)

        return tokenized_training_examples


    def add_architecture_special_tokens(self):
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary must exist before adding special tokens.")

        # get the list of default special tokens for this architecture
        architecture_special_tokens = DEFAULT_SPECIAL_TOKENS[self.architecture]

        # all the special tokens are the default-arch-special-tokens plus the additional special tokens
        all_special_tokens = (architecture_special_tokens + self.additional_special_tokens)

        # iterate all special tokens and add it to the vocabulary and our other storage of special tokens
        for special_token in all_special_tokens:
            special_token_id = self.vocabulary.add_special_token(special_token)
            self.special_token_ids[special_token] = special_token_id

    
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

    def get_special_token_id(self, token: str) -> int:
        self.check_is_trained()

        if token not in self.special_token_ids:
            raise KeyError(f"Special token {token!r} is not configured for "f"{self.architecture.value!r}.")

        return self.special_token_ids[token]

    def get_special_token_ids(self) -> dict[str, int]:
        self.check_is_trained()
        return self.special_token_ids.copy()

    def check_is_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "The tokenizer must be trained before calling "
                "tokenize(), encode(), or decode()."
            )

# run: python -m tokenization.tokenizer, library/
if __name__ == "__main__":

    print("------ Testing Complete BPE Tokenizer ------\n")

    corpus = [
        "hello",
        "help",
        "helmet",
    ]

    tokenizer = BPETokenizer(
        architecture=TransformerArchitecture.ENCODER_DECODER,
    )

    tokenizer.train(
        corpus=corpus,
        target_vocab_size=10,
    )

    original_text = "hello"

    tokens = tokenizer.tokenize(original_text)
    token_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(token_ids)

    print("Architecture:")
    print(tokenizer.architecture.value)

    print("\nVocabulary:")
    print(tokenizer.get_vocabulary())

    print("\nMerge Rules:")
    print(tokenizer.get_merge_rules())

    print("\nSpecial Token IDs:")
    print(tokenizer.get_special_token_ids())

    print("\nPAD ID:")
    print(tokenizer.get_special_token_id("<PAD>"))

    print("\nBOS ID:")
    print(tokenizer.get_special_token_id("<BOS>"))

    print("\nEOS ID:")
    print(tokenizer.get_special_token_id("<EOS>"))

    print("\nUNK ID:")
    print(tokenizer.get_special_token_id("<UNK>"))

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

    print("\nFinal Vocabulary Size:")
    print(tokenizer.get_vocab_size())

    print("\n------ Testing Encode Training Examples ------\n")     # note: this test has to match the vocabuary above
    training_examples = [
        EncoderDecoderTextTrainingExample(
            source_text="helo",
            target_text="hello",
        ),
        EncoderDecoderTextTrainingExample(
            source_text="help",
            target_text="helmet",
        ),
    ]

    encoded_training_examples = tokenizer.encode_training_examples(
        training_examples=training_examples
    )

    print("Number of text training examples:")
    print(len(training_examples))

    print("\nNumber of encoded training examples:")
    print(len(encoded_training_examples))

    print("\nText Training Example 1:")
    print("source: " + training_examples[0].source_text)
    print("target: " + training_examples[0].target_text)
    print("\nText Training Example 2:")
    print("source: " + training_examples[1].source_text)
    print("target: " + training_examples[1].target_text)

    for indx, encoded_example in enumerate(encoded_training_examples):

        print(f"\nEncoded Training Example {indx + 1}:")

        print("Source Token IDs:")
        print(encoded_example.source_token_ids)

        print("Target Token IDs:")
        print(encoded_example.target_token_ids)

    print("\nOne encoded example generated per text training example:")
    print(len(encoded_training_examples) == len(training_examples))