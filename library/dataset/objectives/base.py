"""
FILE: defines the base interface for training objectives that turn unpaired text into encoder-decoder source-target training examples.

"""

from abc import ABC, abstractmethod
from random import Random
from dataset.training_example import EncoderDecoderTextTrainingExample

class UnpairedTextTrainingObjectiveBase(ABC):

    @abstractmethod
    def create_examples(self, text: str, random_generator: Random) -> list[EncoderDecoderTextTrainingExample]:
        """
        Converts one raw text item into one or more training examples. 
        Returns onoe or more EncoderDecoderTextTrainingExample-objs
        """

        return NotImplementedError