"""
FILE: this is one of the training objectives for that creates training text example from unpaired text. You can implement others nad your own. 
This is the default for unpaired text. Step 4. 

Denoising Autoencoding works by taking each text item in the unpaired corpus and modifying it in some random way to
get the source-corrupted-sequence and the target-sequence is the original untampered sequence, this is one training pair. You do this with all items in the text corpus to get all training examples.
"""


from dataset.training_example import EncoderDecoderTextTrainingExample
from dataset.objectives.base import UnpairedTextTrainingObjectiveBase
from random import Random


class DenoisingAutoencodingTrainingOjective(UnpairedTextTrainingObjectiveBase): # inherits from base training objective

    def __init__(self, word_deletion_prob: float = 0.15, character_deletion_prob: float = 0.15):
        self.word_deletion_prob = word_deletion_prob                # probability of each word being deleted in order to create the corrputed sequence      
        self.character_deletion_prob = character_deletion_prob      # probability of each char being deleted in order to crete the corrputed sequence

    """
    Creates one training example of the form source-corrupted-sequence -> target-original-sequence using denoising autoencoding from one given text item from unpaired corpus. 
    Then this is applied to all items in unpaired corpus to get all text training examples.
    """
    def create_examples(self, text: str, random_generator: Random) -> list[EncoderDecoderTextTrainingExample]:
        # split the text into words so we can determine whether this corpus itme contains multiple words or only one word
        words = text.split()

        # if the text contains multiple words, get the source-corrupted-text by deleting words
        if len(words) > 1:
            corrupted_text = self.delete_words(words=words, random_generator=random_generator)
        # if the text contains only one word, get the source-corrupted-text by deleting characters
        else:
            corrupted_text = self.delete_characters(text=text, random_generator=random_generator)

        # create hte encoder-decoder training example, where the source-text is the corrupted-text & target-text is the original-text, this is denoising-autoencoding
        training_example = EncoderDecoderTextTrainingExample(source_text=corrupted_text, target_text=text)

        # return a list because a training objective may eventually be able to create multiple training examples from one corpus item
        return [training_example]


    """
    Given list of words strings, delete words randomly while guaranteeing that:
    1. At least one word is deleted
    2. At least one word remains
    """
    def delete_words(self, words: list[str], random_generator: Random) -> str:
        # for every word in words get a T/F if we should delete it, if a random generated num between 0.0 & 1.0 is greater than equal to the probability of a word being deleted
        keep_word = [random_generator.random() >= self.word_deletion_prob for _ in words] # T = keep, F = delete

        # if it turned out that all are T, then all words should be kept, guarntee that corrpution actually occurs, force one word to be deleted
        if all(keep_word):
            deletion_indx = random_generator.randrange(len(words)) # pick random word-indx to delete
            keep_word[deletion_indx] = False                       # mark that word's bool as should be deleted

        # if all words are F then all words are deleted, so prevent the corruped source from becoming empty
        if not any(keep_word):
            keep_indx = random_generator.randrange(len(words))      # pick random word-indx to keep, so make sure at least one word is kept
            keep_word[keep_indx] = True                             # mark that word's bool as should be kept


        # for word and its bool on weather we should keep it if we should add to list of corrupted words strings
        corrupted_words = []
        for word, should_keep in zip(words, keep_word, strict=True):
            if should_keep:
                corrupted_words.append(word)

        return " ".join(corrupted_words)        # joing all words back seprated by space

    """
    Given a word-string delete characters randomly in it, For single word corpus item, while guaranteeing that:
    1.  At least one character is deleted.
    2.  At least one character remains
    """
    def delete_characters(self, text: str, random_generator: Random) -> str:
        # create list of characters from the given word-string-item
        characters = list(text)

        # if the number of chars in the word is less than or equal to one you cannot delete a char from it without making the source-corrupted-sequence empty
        if len(characters) <= 1:
            return text

        # for char in word get a T/F if we shouuld delete it or not, if a random generated num between 0.0 & 1.0 is greater than equal to the probability of a char being deleted
        keep_character = [ random_generator.random() >= self.character_deletion_prob for _ in characters ] # T = keep, F = delete

        # if all chars are T means kept, then  guarantee that at least one character is deleted
        if all(keep_character):
            deletion_indx = random_generator.randrange(len(characters)) # pick ranomd char-indx to delete
            keep_character[deletion_indx] = False                      # mark that char's bool as should be deleted

        # if all chars  are F means deleted, then guarantee that at least one char is kept
        if not any(keep_character):
            keep_indx = random_generator.randrange(len(characters))
            keep_character[keep_indx] = True

        # for every char and based its bool to keep or not, add it to the list of corrupted chars
        corrupted_chars = []
        for char, should_keep in zip(characters, keep_character, strict=True):
            if should_keep:
                corrupted_chars.append(char)

        return "".join(corrupted_chars)        # joining all chars contiguously

# run: python -m dataset.objectives.denoising, library/
if __name__ == "__main__":

    print("------ Testing Denoising Autoencoding Objective ------\n")

    objective = DenoisingAutoencodingTrainingOjective(
        word_deletion_prob=0.30,
        character_deletion_prob=0.30,
    )

    random_generator = Random(42)    # fixed seed so the test gives reproducible random results
    random_generator = Random()      # not fixed seed to see different results everytime

    # --------------------------------------------------
    # Test 1: multi-word text -> delete words
    # --------------------------------------------------
    print("\n---------------TEST 1-----------------------")

    text = "The quick brown fox jumps over the lazy dog"
    words = text.split()
    corrupted_text = objective.delete_words(words=words, random_generator=random_generator)


    print("\nSource Corrupted multi-word text:")
    print(corrupted_text)
    
    print("\nTarget Original multi-word text:")
    print(text)


    print("\nAt least one word was deleted:")
    print(corrupted_text != text)

    print("\nCorrupted text is not empty:")
    print(len(corrupted_text) > 0)

    # --------------------------------------------------
    # Test 2: single word -> delete characters
    # --------------------------------------------------

    word = "hello"
    corrupted_word = objective.delete_characters(text=word, random_generator=random_generator)

    print("\n---------------TEST 2-----------------------")

    #
    print("\nSource Corrupted single word:")
    print(corrupted_word)

    print("\nTarget Original single word:")
    print(word)

    print("\nAt least one character was deleted:")
    print(corrupted_word != word)

    print("\nCorrupted word is not empty:")
    print(len(corrupted_word) > 0)