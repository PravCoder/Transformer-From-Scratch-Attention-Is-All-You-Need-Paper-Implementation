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
    Construct model input and target sequences (model training examples)
"""

from collections.abc import Iterable
from tokenization.tokenizer import BPETokenizer
from tokenization.special_tokens import TransformerArchitecture
from dataset.paired_example_builder import PairedTextTrainingExampleBuilder
from dataset.unpaired_example_builder import UnpairedTextTrainingExampleBuilder
from dataset.objectives.base import UnpairedTextTrainingObjectiveBase
from dataset.training_example import EncoderDecoderTextTrainingExample, EncoderDecoderTokenizedTrainingExample , EncoderDecoderModelTrainingExample

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
    def create_dataset(self, corpus: Iterable[tuple[str, str]] | Iterable[str], debug: bool = False):
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

        self.debug_step4(debug=debug, training_examples=text_training_examples)

        # ================ STEP 5: train tokenizer ================
        
        # defines the corpus used to train the tokenizer
        tokenizer_corpus = corpus

        # for paired-text the we have to put all the pairs and its source/target text into one corpus to create the corpus that is used to train the tokenizer
        if isinstance(corpus[0], tuple):
            tokenizer_corpus = self.create_corpus_for_tokenizer(text_training_examples)     

        # for unpaired-text the unpaired-corpus is just the corpus used to train the tokenizer
        # use the tokenizer-obj to train the tokenizer which creates the vocab & merge rules
        self.tokenizer.train(corpus=tokenizer_corpus, target_vocab_size=self.target_vocab_size)

        self.debug_step5_tokenizer(debug=debug)
        

        # ================ STEP 5: Encode the text training examples ================
        # after training the tokenizer, use it to encode the text-training-examples into token IDs, gives list of EncoderDecoderTokenizedTrainingExample-objs
        tokenized_training_examples = self.tokenizer.encode_training_examples(text_training_examples)

        self.debug_step5_encoding(debug=debug, training_examples=tokenized_training_examples)

        # ================ STEP 6: Construct model input and target sequences (model training examples) ================
        model_training_examples = self.construct_model_sequences(tokenized_training_examples)

        self.debug_step6(debug=debug, training_examples=model_training_examples)

        return model_training_examples



    """
    Step 6: Construct model input and target sequences example objs for encoder-decoder, given the tokenized training examples.
    
    Tokenized Example:
        source_ids = [s1, s2, s3]
        target_ids = [t1, t2, t3]
    
    Model Training Example:
        encoder_input = [s1, s2, s3, EOS]
        decoder_input = [BOS, t1, t2, t3]
        decoder_target = [t1, t2, t3, EOS]
    """
    def construct_model_sequences(self, tokenized_training_examples: list[EncoderDecoderTokenizedTrainingExample]) -> list[EncoderDecoderModelTrainingExample]:

        bos_token_id = self.tokenizer.get_special_token_id("<BOS>")     # get the beginning-of-sentence special-token-id
        eos_token_id = self.tokenizer.get_special_token_id("<EOS>")     # get the end-of-sentence special-token-id

        # stores the created model-training-example-objs
        model_training_examples: list[ EncoderDecoderModelTrainingExample ] = []
        
        # for every tokenized-training-example-obj, construct the model sequences
        for cur_tokenized_example in tokenized_training_examples:
            # get the cur-tokenized-example source-token-ids and add EOS-token to the end of it
            encoder_input_ids = cur_tokenized_example.source_token_ids + [eos_token_id]
            # get the cur-tokenized-example target-token-ids and add BOS to the start of it
            decoder_input_ids = [bos_token_id] + cur_tokenized_example.target_token_ids
            # get the cur-tokenized-example  target-token-ids and add EOS to the end of it, this does the "one token shift", so the model is he model is trained to predict the next correct token.
            decoder_target_ids = cur_tokenized_example.target_token_ids + [eos_token_id]

            # create the model-training-example by combining all 3 sequences for encoder-deocder
            cur_model_training_example = EncoderDecoderModelTrainingExample(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, decoder_target_ids=decoder_target_ids)

            model_training_examples.append(cur_model_training_example)

        return model_training_examples





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


    # ============================== DEBUG HELPERS ==============================
    def debug_step4(self, debug: bool, training_examples: list[EncoderDecoderModelTrainingExample]) -> None:
        if not debug:
            return

        print("\n=============== STEP 4 ===============")
        print("Created Text Training Examples")

        for indx, training_example in enumerate(training_examples):

            print(f"\nTraining Example {indx + 1}")
            print(training_example)

    
    def debug_step5_tokenizer(self, debug: bool) -> None:
        if not debug:
            return

        print("\n=============== STEP 5: TOKENIZER ===============")

        print("\nVocabulary:")
        print(self.tokenizer.get_vocabulary())

        print("\nMerge Rules:")
        print(self.tokenizer.get_merge_rules())

        print("\nSpecial Token IDs:")
        print(self.tokenizer.get_special_token_ids())

        print("\nVocabulary Size:")
        print(self.tokenizer.get_vocab_size())

    def debug_step5_encoding(self, debug: bool, training_examples: list[EncoderDecoderTokenizedTrainingExample]) -> None:
        if not debug:
            return
        print(
            "\n=============== STEP 5: ENCODE TRAINING EXAMPLES ==============="
        )
        for indx, training_example in enumerate(training_examples):

            print(f"\nTraining Example {indx + 1}")
            print(training_example)

    def debug_step6(self, debug: bool, training_examples: list[EncoderDecoderModelTrainingExample]) -> None:
        if not debug:
            return

        print("\n=============== STEP 6: CONSTRUCT MODEL SEQUENCES ===============")
        for indx, training_example in enumerate(training_examples):

            print(f"\nTraining Example {indx + 1}")
            print(training_example)

    



# run: python -m dataset.dataset_creation_pipeline, library/
if __name__ == "__main__":

    print(
        "\n------ Testing Encoder-Decoder Dataset Creation Pipeline ------"
    )

    # ============================================================
    # TEST 1: PAIRED CORPUS
    # ============================================================

    print(
        "\n\n================ TEST 1: PAIRED CORPUS ================"
    )

    paired_corpus = [
        ("hello", "help"),
        ("help", "helmet"),
    ]

    print("\nOriginal Paired Corpus:")

    for indx, pair in enumerate(paired_corpus):
        source_text, target_text = pair

        print(f"\nCorpus Item {indx + 1}")
        print(f"Source: {source_text}")
        print(f"Target: {target_text}")

    # create pipeline
    paired_pipeline = EncoderDecoderDatasetCreationPipeline(
        target_vocab_size=10,
    )

    # run entire dataset pipeline so far
    paired_dataset = paired_pipeline.create_dataset(
        corpus=paired_corpus,
        debug=True,
    )

    # final output of the pipeline so far
    print(
        "\n=============== FINAL PAIRED DATASET OUTPUT ==============="
    )

    for indx, training_example in enumerate(paired_dataset):
        print(f"\nTraining Example {indx + 1}")
        print(training_example)


    # ============================================================
    # TEST 2: UNPAIRED CORPUS
    # ============================================================

    print(
        "\n\n================ TEST 2: UNPAIRED CORPUS ================"
    )

    unpaired_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Transformers learn representations from text",
        "hello",
    ]

    print("\nOriginal Unpaired Corpus:")

    for indx, text in enumerate(unpaired_corpus):
        print(f"\nCorpus Item {indx + 1}")
        print(text)

    # create a new pipeline
    # training_objective=None means use default denoising autoencoding
    unpaired_pipeline = EncoderDecoderDatasetCreationPipeline(
        target_vocab_size=30,
        training_objective=None,
        random_seed=42,
    )

    # run entire dataset pipeline so far
    unpaired_dataset = unpaired_pipeline.create_dataset(
        corpus=unpaired_corpus,
        debug=True,
    )

    # final output of the pipeline so far
    print(
        "\n=============== FINAL UNPAIRED DATASET OUTPUT ==============="
    )

    for indx, training_example in enumerate(unpaired_dataset):
        print(f"\nTraining Example {indx + 1}")
        print(training_example)