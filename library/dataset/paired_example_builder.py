"""
FILE: this is step 4 case 1, it creates the training-examples from the text corpus when the corpus is given as pairs of source and target sequences

"""

from collections.abc import Iterable
from dataset.training_example import EncoderDecoderTextTrainingExample

"""
Creates training examples from data that already contains source-target pairs
"""
class PairedTextTrainingExampleBuilder:

    """
    This builds the pairs when the paired corpus is a list of tuples:
        pairs = [
                    ("Hello", "Bonjour"),
                    ("Goodbye", "Au revoir"),
                ]
    """
    def build_from_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[EncoderDecoderTextTrainingExample]:

        # list of the training example created from paired corpus
        training_examples: list[EncoderDecoderTextTrainingExample] = []
        # iterate all text pairs in given paired corpus
        for indx, pair in enumerate(pairs):

            source_text, target_text = pair
            # construct a training-example for the current tuple text pair
            cur_example = EncoderDecoderTextTrainingExample(source_text=source_text, target_text=target_text)

            training_examples.append(cur_example)

        # return all training-example-objs
        return training_examples


# run: python -m dataset.paired_example_builder, library/
if __name__ == "__main__":
    paired_corpus = [
        ("Hello", "Bonjour"),
        ("Goodbye", "Au revoir"),
    ]

    builder = PairedTextTrainingExampleBuilder()

    training_examples = builder.build_from_pairs(paired_corpus)

    for example in training_examples:
        print(example)