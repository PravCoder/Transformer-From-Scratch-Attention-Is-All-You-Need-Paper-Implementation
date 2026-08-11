"""
FILE: this is step 4 case 2, it creates the training examples from the text corpus when it is given as unpaired text corpus
"""

from dataset.objectives.base import UnpairedTextTrainingObjectiveBase
from dataset.objectives.denoising import DenoisingAutoencodingTrainingOjective
from dataset.training_example import EncoderDecoderTextTrainingExample
from collections.abc import Iterable
from random import Random



"""
Given an unpaired corpus creates training examples from it using a training objective.

Applies a training objective to very item in an unpaired corpus. If no objective is supplied denosining autoencoding is used. 
"""
class UnpairedTextTrainingExampleBuilder:

    def __init__(self, training_objective: UnpairedTextTrainingObjectiveBase, random_seed: int | None = 42) -> None:

        # if no custom training objective is given use default denosining autoencoding
        if training_objective is None:
            training_objective = DenoisingAutoencodingTrainingOjective()


        self.training_objective = training_objective
        self.random_seed = random_seed

    """
    Given a corpus made up of "corpus items" strings. 
    Apply the selected training objective to the entire unpaired corpus which creates training examples from the unpaired corpus.
    Returns a list of training-examples-objs.
    """
    def build_training_examples(self, corpus: Iterable[str]) -> list[EncoderDecoderTextTrainingExample]:

        random_generator = Random(self.random_seed)

        # stores training-examples-objs
        training_examples: list[ EncoderDecoderTextTrainingExample ] = []

        # iterate every corpus-item in unpaired-corpus
        for indx, cur_corpus_item in enumerate(corpus):
            # for the cur-forpus-item use the training-objective to create a training-example from it, returns one or more trainnig-example-obj, look at denosing.py
            generated_training_examples = self.training_objective.create_examples(text=cur_corpus_item, random_generator=random_generator)

            # add the created training-example from this corpus-item to all created training-example-objs
            training_examples.extend(generated_training_examples)

        # return all created training-example from te unpaired corpus
        return training_examples

# run: python -m dataset.unpaired_example_builder, library/
if __name__ == "__main__":

    print("------ Testing Unpaired Text Training Example Builder ------\n")

    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Transformers learn representations from text",
        "hello",
    ]

    # --------------------------------------------------
    # TEST 1: Default training objective
    # --------------------------------------------------

    print("--------------- TEST 1: DEFAULT DENOISING ---------------\n")

    builder = UnpairedTextTrainingExampleBuilder(
        training_objective=None,
        random_seed=None,
    )

    training_examples = builder.build_training_examples(
        corpus=corpus
    )

    print("Number of corpus items:")
    print(len(corpus))

    print("\nNumber of generated training examples:")
    print(len(training_examples))

    print("Created Training Examples:")
    for indx, example in enumerate(training_examples):

        print(f"\nTraining Example {indx + 1}:")

        print("Source:")
        print(example.source_text)

        print("Target:")
        print(example.target_text)

    print("\nOne training example generated per corpus item:")
    print(len(training_examples) == len(corpus))

    # --------------------------------------------------
    # TEST 2: Verify target remains original corpus item
    # --------------------------------------------------

    print("\n--------------- TEST 2: TARGETS ---------------\n")

    for original_text, example in zip(
        corpus,
        training_examples,
        strict=True
    ):
        print("Target equals original corpus item:")
        print(example.target_text == original_text)

    # --------------------------------------------------
    # TEST 3: Verify source was corrupted
    # --------------------------------------------------

    print("\n--------------- TEST 3: CORRUPTION ---------------\n")

    for original_text, example in zip(
        corpus,
        training_examples,
        strict=True
    ):
        print(f"Original:  {original_text}")
        print(f"Corrupted: {example.source_text}")
        print(
            "Source differs from target:",
            example.source_text != example.target_text
        )
        print()