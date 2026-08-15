"""
FILE: the dataset creation pipeline connects together all the componenets in dataset/. For 

Currently implemented:

Step 4:
    Create Training Text Examples from Text Corpus
        Case 1: Paired Corpus
        Case 2: Unpaired Corpus

Step 5:
    Build or train the tokenizer and vocabulary.
        - Train BPE - Build vocabulary - Encode - Decode - Add Required special tokens to vocab - Encode the text training examples

Step 6:
    Construct model input and target sequences
"""

from collections.abc import Iterable
from tokenization.tokenizer import BPETokenizer
from tokenization.special_tokens import TransformerArchitecture
from dataset.paired_example_builder import PairedTextTrainingExampleBuilder
from dataset.unpaired_example_builder import UnpairedTextTrainingExampleBuilder
from dataset.objectives.base import UnpairedTextTrainingObjectiveBase
from dataset.training_example import EncoderDecoderTextTrainingExample, EncoderDecoderTokenizedTrainingExample 

class EncoderDecoderDatasetCreationPipeline:

    def __init__(self, target_vocab_size: int, training_objective: UnpairedTextTrainingObjectiveBase | None = None, random_seed: int | None = 42) -> None:

        self.target_vocab_size = target_vocab_size          # the size of the vocabulary you want to reach when training tokenizer
        self.training_objective = training_objective        # defines how text examples are created from unpaired-corpus
        self.random_seed = random_seed

        # define tokenizer-obj
        self.tokenizer = BPETokenizer(architecture=TransformerArchitecture.ENCODER_DECODER)

    """
    Builds the dataset fed into the model given either paired or unpaired corpus text, by following the steps above.
    """
    def create_dataset(self, corpus: Iterable[tuple[str, str]] | Iterable[str]):
        # convert iterable into a list
        corpus = list(corpus)

        # ================ STEP 4 CASE 1: corpus is given as source-target pairs ================
        if isinstance(corpus[0], tuple):
            # create paired-example-builder-obj which has the tool for creating examples from this paired text
            example_builder = PairedTextTrainingExampleBuilder()

            # creates EncoderDecoderTextTrainingExample-objs from either example builder, paired-corpus  just copy it into objs, 
            text_training_examples = example_builder.build_from_pairs(corpus)     # these are text examples

        # ================ STEP 4 CASE 2: corpus is given as unpaired text ================
        if isinstance(corpus[0], str):
            # create unpaired-example-builder-obj which defines the training objective which defines how to create examples from that unpaired text
            example_builder = UnpairedTextTrainingExampleBuilder(training_objective=self.training_objective, random_seed=self.random_seed)

            # unpaired-corpus so uses the training-objective defined to create text-example-objs EncoderDecoderTextTrainingExample
            text_training_examples = example_builder.build_training_examples(corpus)
        

        

        # ================ STEP 5: train tokenizer ================
        
        # defines the corpus used to train the tokenizer
        tokenizer_corpus = corpus

        # for paired-text the we have to put all the pairs and its source/target text into one corpus to create the corpus that is used to train the tokenizer
        if isinstance(corpus[0], tuple):
            tokenizer_corpus = self.create_corpus_for_tokenizer(text_training_examples)     

        # for unpaired-text the unpaired-corpus is just the corpus used to train the tokenizer
        # use the tokenizer-obj to train the tokenizer which creates the vocab & merge rules
        self.tokenizer.train(corpus=tokenizer_corpus, target_vocab_size=self.target_vocab_size)

        # ================ STEP 5: Encode the text training examples ================
        # after training the tokenizer use to encode the text-training-examples into token IDs, gives list of EncoderDecoderTokenizedTrainingExample-objs
        tokenized_training_examples = self.tokenizer.encode_training_examples(text_training_examples)

        return tokenized_training_examples



              



    """
    This creates the corpus needed to train the tokenizer given the text-training-examples-objs. Only used for paired corpus.
    """
    def create_corpus_for_tokenizer(self, training_examples: list[EncoderDecoderTextTrainingExample]) -> list[str]:
        # the corpus needed to train the tokenizer is just a list of strings
        tokenizer_corpus: list[str] = []

        # for every training-example-obj
        for training_example in training_examples:
            # add the source-text to the tokenizer-corpus
            tokenizer_corpus.append(training_example.source_text)
            # add the target-text to the tokenizer-corpus
            tokenizer_corpus.append(training_example.target_text)

        return tokenizer_corpus



# run: python -m dataset.dataset_creation_pipeline, library/
if __name__ == "__main__":

    print(
        "------ Testing Encoder-Decoder Dataset Creation Pipeline ------"
    )

    # ============================================================
    # TEST 1: PAIRED CORPUS
    # ============================================================

    print("\n=============== TEST 1: PAIRED CORPUS ===============\n")

    paired_corpus = [
        ("hello", "help"),
        ("help", "helmet"),
    ]

    paired_pipeline = EncoderDecoderDatasetCreationPipeline(
        target_vocab_size=10,
    )

    paired_tokenized_examples = (
        paired_pipeline.create_dataset(
            paired_corpus
        )
    )

    print("Original paired corpus:")

    for source, target in paired_corpus:
        print(f"Source: {source}")
        print(f"Target: {target}")
        print()

    print("Tokenized training examples:")

    for indx, example in enumerate(
        paired_tokenized_examples
    ):
        print(f"\nTraining Example {indx + 1}")

        print("Source Token IDs:")
        print(example.source_token_ids)

        print("Target Token IDs:")
        print(example.target_token_ids)

    print("\nNumber of original pairs:")
    print(len(paired_corpus))

    print("\nNumber of tokenized training examples:")
    print(len(paired_tokenized_examples))

    print("\nOne training example per pair:")
    print(
        len(paired_corpus)
        == len(paired_tokenized_examples)
    )

    print("\nTokenizer vocabulary:")
    print(
        paired_pipeline.tokenizer.get_vocabulary()
    )

    # ============================================================
    # TEST 2: UNPAIRED CORPUS
    # ============================================================

    print(
        "\n=============== TEST 2: UNPAIRED CORPUS ===============\n"
    )

    unpaired_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Transformers learn representations from text",
        "hello",
    ]

    unpaired_pipeline = EncoderDecoderDatasetCreationPipeline(
        target_vocab_size=30,
        random_seed=42,
    )

    unpaired_tokenized_examples = (
        unpaired_pipeline.create_dataset(
            unpaired_corpus
        )
    )

    print("Original unpaired corpus:")

    for text in unpaired_corpus:
        print(text)

    print("\nTokenized training examples:")

    for indx, example in enumerate(
        unpaired_tokenized_examples
    ):
        print(f"\nTraining Example {indx + 1}")

        print("Source Token IDs:")
        print(example.source_token_ids)

        print("Target Token IDs:")
        print(example.target_token_ids)

    print("\nNumber of corpus items:")
    print(len(unpaired_corpus))

    print("\nNumber of tokenized training examples:")
    print(len(unpaired_tokenized_examples))

    print("\nOne denoising example per corpus item:")
    print(
        len(unpaired_corpus)
        == len(unpaired_tokenized_examples)
    )

    print("\nTokenizer vocabulary size:")
    print(
        unpaired_pipeline.tokenizer.get_vocab_size()
    )


        





