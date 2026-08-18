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

Step 7:
    Pad or truncate Sequences and Create Masks
        Part 1: Choose Maximum Sequence Lengths
        Part 2: Truncate Sequences that are too long
        Part 3: Pad Sequences that are too short
        Part 4: Create Padding Masks
        Part 5: Create Causal Mask

Step 8:
    Batch the Examples

Step 9:
    Convert batches into tensor and prepare model inputs
"""

from collections.abc import Iterable
import torch
from tokenization.tokenizer import BPETokenizer
from tokenization.special_tokens import TransformerArchitecture
from dataset.paired_example_builder import PairedTextTrainingExampleBuilder
from dataset.unpaired_example_builder import UnpairedTextTrainingExampleBuilder
from dataset.objectives.base import UnpairedTextTrainingObjectiveBase
from dataset.training_example import EncoderDecoderTextTrainingExample, EncoderDecoderTokenizedTrainingExample , EncoderDecoderModelTrainingExample, EncoderDecoderTrainingBatch

class EncoderDecoderDatasetCreationPipeline:

    def __init__(self, target_vocab_size: int, max_encoder_length: int, max_decoder_length: int, batch_size: int, device: str = "cpu", training_objective: UnpairedTextTrainingObjectiveBase | None = None, random_seed: int | None = 42) -> None:

        self.target_vocab_size = target_vocab_size          # the size of the vocabulary you want to reach when training tokenizer
        self.training_objective = training_objective        # defines how text examples are created from unpaired-corpus
        self.random_seed = random_seed
        self.device = device

        self.max_encoder_length = max_encoder_length        # the max length each encoder sequence must be
        self.max_decoder_length = max_decoder_length        # the max length each deocder sequence must be

        self.batch_size = batch_size                        # the number of training-examples each batch should have

        # create causal mask which is 2D lower triangular matrix, because it only depends on max decoder, so only create it once. 
        self.decoder_causal_mask = self.create_causal_mask(self.max_decoder_length)  # this is shared across all examples  because it only depends on decoder sequence max length, which is constant across all examples
        
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

        # ================ STEP 7: Pad or truncate Sequences and Create Masks ================
        model_training_examples = self.pad_truncate_and_create_masks(model_training_examples)

        self.step7_debug(debug=debug,training_examples=model_training_examples)

        # ================ STEP 8: Batch the Examples ================
        training_batches = self.batch_training_examples(model_training_examples)        # returns list of training-batch-objs
        self.step8_debug(debug=debug,training_batches=training_batches)

        # ================ STEP 9: Convert Batch to Tensors and prepare model inputs ================
        training_batches = self.convert_batches_to_tensors(training_batches)
        # covnert shared var to tensor as well
        self.decoder_causal_mask = torch.tensor(self.decoder_causal_mask, dtype=torch.bool, device=self.device)

        self.step9_debug(debug=debug,tensor_batches=training_batches)


        return model_training_examples



    """
    Step 6: Construct model input and target sequences example objs for encoder-decoder, given the tokenized training examples.
    
    Given Tokenized Example:
        source_ids = [s1, s2, s3]
        target_ids = [t1, t2, t3]
    
    Output Model Training Example:
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
    Step 7: Pad or Truncate Sequences and Create Masks.
    For each model training example and all three of its sequences make sure they are the same length of either the defined max_encoder_length or max_decoder_length.
    Then create padding masks to ignore <PAD> tokens and causal masks to prevent decoder from seeing future tokens.
    """
    def pad_truncate_and_create_masks(self, training_examples: list[EncoderDecoderModelTrainingExample]) -> list[EncoderDecoderModelTrainingExample]:

        pad_token_id = self.tokenizer.get_special_token_id("<PAD>")
        eos_token_id = self.tokenizer.get_special_token_id("<EOS>")
        bos_token_id = self.tokenizer.get_special_token_id("<BOS>")


        # iterate every model-training-example-obj
        for cur_training_example in training_examples:

            # --------------- PART 2: Truncate Sequences That Are Too Long ---------------

            # truncate encoder-input-ids-sequence if needed, pass in encoder-input-sequence & max-encoder-length
            cur_training_example.encoder_input_ids = self.truncate_encoder_sequence(sequence=cur_training_example.encoder_input_ids, max_length=self.max_encoder_length, eos_token_id=eos_token_id)

            # truncate decoder-sequences if needed, pass in both decoder-input-sequence & decoder-target-sequence, max-decoder-length
            truncated_decoder_input_ids, truncated_decoder_target_ids = self.truncate_decoder_sequences(decoder_input_ids=cur_training_example.decoder_input_ids, decoder_target_ids=cur_training_example.decoder_target_ids, max_length=self.max_decoder_length, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

            # set the truncated decoder sequences of this model-training-example
            cur_training_example.decoder_input_ids = truncated_decoder_input_ids
            cur_training_example.decoder_target_ids = truncated_decoder_target_ids

            # --------------- PART 3: Pad Sequences That Are Too Short ---------------
            # simply pad all three of the sequences with <PAD> for cur-model-training-example so it reaches the required respective sequence-max-length
            cur_training_example.encoder_input_ids = self.pad_sequence(sequence=cur_training_example.encoder_input_ids, max_length=self.max_encoder_length, pad_token_id=pad_token_id)

            cur_training_example.decoder_input_ids = self.pad_sequence(sequence=cur_training_example.decoder_input_ids, max_length=self.max_decoder_length, pad_token_id=pad_token_id)

            cur_training_example.decoder_target_ids = self.pad_sequence(sequence=cur_training_example.decoder_target_ids, max_length=self.max_encoder_length, pad_token_id=pad_token_id)

            # --------------- PART 4: Create Padding Masks ---------------
            # padding masks need to be created for the encoder and decoder, these mask the encoder-input-ids & decoder-input-ids respectively
            cur_training_example.encoder_padding_mask = self.create_padding_mask(sequence=cur_training_example.encoder_input_ids, pad_token_id=pad_token_id)

            cur_training_example.decoder_padding_mask = self.create_padding_mask(sequence=cur_training_example.decoder_input_ids, pad_token_id=pad_token_id)


            # --------------- PART 5: Create Casual Mask ---------------
            # is already done during the pipeline initiation, 

        return training_examples

    """
    Step-7 Part-2: Truncate Sequences That Are Too Long. For encoder sequences.
    Args:
        sequence: the sequence of token-ids that maybe too long that we need to truncate.
        max_length: the maximum length the encoder sequence can be 
        eos_token_id: since it is a encoder-sequence we may have to insert <EOS> token
    """
    def truncate_encoder_sequence(self, sequence: list[int], max_length: int, eos_token_id: int) -> list[int]:
        # if the length of the encoder-sequence is already less than or equal to the encoder-max-length then 
        if len(sequence) <= max_length:
            return sequence

        # the truncated sequence is from the first position to just before the last max length position
        truncated_sequence = sequence[:max_length]
        # set the last max length position to be the <EOS> token
        truncated_sequence[-1] = eos_token_id

        return truncated_sequence

    """
        Step-7 Part-2: Truncate Sequences That Are Too Long. For decoder sequences (input-sequence & target-sequence)
        Args:
            decoder_input_ids: the sequence of token-ids for decoder input
            decoder_target_ids: the sequence of token-ids for decoder target
            max_length: the maximum length the decoder sequence can be 
            eos_token_id: since it is a decoder-sequence we may have to insert <EOS> token
            bos_token_id: since it is decoder-sequence we may bave to insert <BOS> token
    """
    def truncate_decoder_sequences(self, decoder_input_ids: list[int], decoder_target_ids: list[int], max_length: int, bos_token_id: int, eos_token_id: int) -> tuple[list[int], list[int]]:
        # if the length of the decoder-input-sequence is already less than or equal to decoder-max-length then no truncation required
        if len(decoder_input_ids) <= max_length:
            return decoder_input_ids, decoder_target_ids

        # truncate both to same size, the max-length-deocder-sequences
        decoder_input_ids = decoder_input_ids[:max_length]
        decoder_target_ids = decoder_target_ids[:max_length]

        # decoder-input-sequence must being with <BOS> token, set start-token to it
        decoder_input_ids[0] = bos_token_id
        # decoder-target-sequence must end with <EOS> token, set end-token to it
        decoder_target_ids[-1] = eos_token_id

        return decoder_input_ids, decoder_target_ids    # reutrn both sequences, we truncate both at same time because they have to be same length.

    """
    Step-7 Part-3: Pad Sequences that are too short.
    Args:
        sequence: a sequence of token-ids from either encoder or decoder
        max_length: the maximum number of tokens this sequence must reach of have
        pad_token_id: token-id of special token <PAD>
    """
    def pad_sequence(self, sequence: list[int], max_length: int, pad_token_id: int) -> list[int]:
        # calculate the number of <PAD> tokens needed for the given sequence to each length of max-length, may be zero if it is already the required max-length
        number_of_padding_tokens = max_length - len(sequence)

        # the padded-sequence is created by taking the original-sequence and adding <PAD> token to the end a certain number of times until it reaches the required length of max-length
        padded_sequence = sequence + ([pad_token_id] * number_of_padding_tokens)

        return padded_sequence
    
    """
    Step-7 Part-4: Create Padding Masks
    Args:
        sequence: a sequence of token-ids that we need to create a padding mask for, from either encoder or decoder.
        pad_token_id: token-id of special token <PAD>
    """
    def create_padding_mask(self, sequence: list[int], pad_token_id: int) -> list[int]:
        # stores the binary padding mask for the given sequence, 1 = real-token & 0 = padding-token
        padding_mask = []
        # iterate every token in sequence
        for token_id in sequence:
            # if that token is a padding-token, add 0 to represent it in the same position in the padding mask
            if token_id == pad_token_id:
                padding_mask.append(0)
            # if that token is not a padding-token, add 1 to represent that it is a real token in the same position in the padding mask.
            else:
                padding_mask.append(1)

        return padding_mask

    """
        Step-7 Part-5: Given the decoder max sequence length it creates the causal mask which is a 2D lower triangular square matrix.
    """
    def create_causal_mask(self, max_sequence_legnth: int) -> list[list[int]]:

        causal_mask = []

        for row in range(max_sequence_legnth):
            mask_row = []

            for column in range(max_sequence_legnth):
                if column <= row:
                    mask_row.append(1)
                else:
                    mask_row.append(0)
            causal_mask.append(mask_row)

        return causal_mask

    """
    Step-8: Batch the Examples
    Args:
        training_examples: takes in the list of EncoderDecoderModelTrainingExample-objs
    Outputs batch-objs
    """
    def batch_training_examples(self, training_examples: list[EncoderDecoderModelTrainingExample]) -> list[EncoderDecoderTrainingBatch]:
        # stores all training-batch-objs
        training_batches: list[ EncoderDecoderTrainingBatch ] = []

        # move through the dataset batch_size examples at a time, batch_start_indx is when the current batch start its index in the model-training-objs list
        for batch_start_indx in range(0, len(training_examples), self.batch_size):

            # the batch-end-indx where the batch ends is where the batch-starts plug the batch-size
            batch_end_indx = batch_start_indx + self.batch_size

            # all of the model-training-examples-objs in our current batch is from the batch'es start-indx to its end-indx
            cur_batch_examples = training_examples[batch_start_indx: batch_end_indx]

            # define all the attributes of a single model-training-obj then we stack them
            # stack encoder input sequences
            batch_encoder_input_ids = []

            # stack decoder input sequences
            batch_decoder_input_ids = []

            # stack decoder target sequences
            batch_decoder_target_ids = []

            # stack encoder padding masks
            batch_encoder_padding_masks = []

            # stack decoder padding masks
            batch_decoder_padding_masks = []

            # iterate every example in our current batch, and at every one of its attribute to into batched-attribute
            for training_example in cur_batch_examples:
                batch_encoder_input_ids.append(training_example.encoder_input_ids)

                batch_decoder_input_ids.append(training_example.decoder_input_ids)

                batch_decoder_target_ids.append(training_example.decoder_target_ids)

                batch_encoder_padding_masks.append(training_example.encoder_padding_mask)

                batch_decoder_padding_masks.append(training_example.decoder_padding_mask)

            # after stacking all attribute create trainin-batch-obj
            training_batch = EncoderDecoderTrainingBatch(
                encoder_input_ids=batch_encoder_input_ids,
                decoder_input_ids=batch_decoder_input_ids,
                decoder_target_ids=batch_decoder_target_ids,
                encoder_padding_mask=batch_encoder_padding_masks,
                decoder_padding_mask=batch_decoder_padding_masks,
            )

            training_batches.append(training_batch)

        return training_batches

    """
    Step 9: Convert batches to tensors and prepare model inputs
    """
    def convert_batches_to_tensors(self, training_batches: list[EncoderDecoderTrainingBatch]) -> list[EncoderDecoderTrainingBatch]:
        # iterate all training-batch-objs created, just convert each attribute in it to a tensor
        for cur_training_batch in training_batches:

            cur_training_batch.encoder_input_ids = torch.tensor(cur_training_batch.encoder_input_ids,dtype=torch.long,device=self.device)

            cur_training_batch.decoder_input_ids = torch.tensor(cur_training_batch.decoder_input_ids,dtype=torch.long,device=self.device,)

            cur_training_batch.decoder_target_ids = torch.tensor(cur_training_batch.decoder_target_ids,dtype=torch.long,device=self.device,)

            cur_training_batch.encoder_padding_mask =  torch.tensor(cur_training_batch.encoder_padding_mask,dtype=torch.bool,device=self.device,)

            cur_training_batch.decoder_padding_mask =  torch.tensor(cur_training_batch.decoder_padding_mask,dtype=torch.bool,device=self.device,)


        return training_batches





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

    def step7_debug(self, debug: bool, training_examples: list[EncoderDecoderModelTrainingExample]) -> None:
        if not debug:
            return

        print("\n=============== STEP 7: ""PAD / TRUNCATE / CREATE MASKS ===============")

        for indx, training_example in enumerate(training_examples):
            print(f"\nTraining Example {indx + 1}")

            print(training_example)

    def step8_debug(self, debug: bool, training_batches: list[EncoderDecoderTrainingBatch]) -> None:
        if not debug:
            return

        print("\n=============== STEP 8: BATCH TRAINING EXAMPLES ===============")
        for indx, training_batch in enumerate(training_batches):

            print(f"\nTraining Batch {indx + 1}")

            print(training_batch)

    def step9_debug(self, debug: bool, tensor_batches: list[EncoderDecoderTrainingBatch]) -> None:
        if not debug:
            return
        print("\n=============== STEP 9: CONVERT BATCHES TO TENSORS ===============")

        for indx, tensor_batch in enumerate(tensor_batches):
            print(tensor_batch)     # note torch may convert some 1/0 ints to true/false that is just how it displays and it makes computation easier
            



# run: python -m dataset.dataset_creation_pipeline, library/
if __name__ == "__main__":

    print(
        "\n------ Testing Encoder-Decoder Dataset Creation Pipeline ------"
    )

    # ============================================================
    # TEST 1: PAIRED CORPUS - TEST PADDING
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
        max_encoder_length=6,
        max_decoder_length=6,
        batch_size=2,
    )

    print("\nMax Encoder Length:")
    print(paired_pipeline.max_encoder_length)
    print("\nMax Decoder Length:")
    print(paired_pipeline.max_decoder_length)

    print("\nDecoder Causal Mask Dimensions:")
    rows = len(paired_pipeline.decoder_causal_mask)
    columns = len(paired_pipeline.decoder_causal_mask[0])
    print(f"{rows} x {columns}")

    print("\nBatch size:")
    print(paired_pipeline.batch_size)

    # run entire dataset pipeline so far
    paired_dataset = paired_pipeline.create_dataset(
        corpus=paired_corpus,
        debug=True,
    )



    # ============================================================
    # TEST 2: UNPAIRED CORPUS - TEST TRUNCATION
    # ============================================================

    print(
        "\n\n================ TEST 2: UNPAIRED CORPUS ================"
    )

    unpaired_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Transformers learn representations from text",
    ]

    print("\nOriginal Unpaired Corpus:")

    for indx, text in enumerate(unpaired_corpus):
        print(f"\nCorpus Item {indx + 1}")
        print(text)

    # create a new pipeline
    # training_objective=None means use default denoising autoencoding
    unpaired_pipeline = EncoderDecoderDatasetCreationPipeline(
        target_vocab_size=30,
        max_encoder_length=10,
        max_decoder_length=10,
        batch_size=2,
        training_objective=None,
        random_seed=42,
    )
    print("\nMax Encoder Length:")
    print(unpaired_pipeline.max_encoder_length)
    print("\nMax Decoder Length:")
    print(unpaired_pipeline.max_decoder_length)

    print("\nDecoder Causal Mask Dimensions:")
    rows = len(unpaired_pipeline.decoder_causal_mask)
    columns = len(unpaired_pipeline.decoder_causal_mask[0])
    print(f"{rows} x {columns}")

    print("\nBatch size:")
    print(unpaired_pipeline.batch_size)

    # run entire dataset pipeline so far
    unpaired_dataset = unpaired_pipeline.create_dataset(
        corpus=unpaired_corpus,
        debug=True,
    )

