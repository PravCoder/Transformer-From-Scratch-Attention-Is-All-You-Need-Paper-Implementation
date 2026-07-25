"""
FILE: implements the BPE Training Algorithm
"""
from tokenization.vocabulary import Vocabulary
from dataclasses import dataclass



# a typed variable to represent two paired tokens
TokenPair = tuple[str, str]
# the current tokenized corpus represents is the original corpus represented using the current vocabulary, i.e, the tokens the tokenizer currently knows. 
# The outer list represents the entire corpus (all text samples), each inner list represents one text sample,  each string inside an inner list is one token
"""
raw corpus:
    corpus = [
        "hello",
        "help",
        "helmet",
    ]

initially tokenized corpus for char-level bpe:
    tokenized_corpus = [
        ["h", "e", "l", "l", "o"],
        ["h", "e", "l", "p"],
        ["h", "e", "l", "m", "e", "t"],
    ]

After Learning ("h", "e") → "he" the tokenized corpus is:
    tokenized_corpus = [
        ["he", "l", "l", "o"],
        ["he", "l", "p"],
        ["he", "l", "m", "e", "t"],
    ]
"""
TOKENIZED_CORPUS = list[list[str]]


"""
Represents everything produced by the BPE tokenizer training. 

"""
@dataclass(frozen=True)
class BPETrainingResult:

    vocabulary: Vocabulary                  # the final trained vocabulary
    merge_rules: list[TokenPair]            # adjacent token pairs in the exact order they were learned, each element is a token-pair that just means these token-pair were merged into one token
    tokenized_corpus: TOKENIZED_CORPUS      # the final tokenized representation of the training corpus


"""
Trains a classic character-level (soon byte-level) Byte Pair Tokenizer

Training Process:
    1. Represent the corpus using the base-level tokens in this case: characters
    2. Add all unqiue base-level tokens (chars) to the initial vocabulary
    3. Count adjacent token pairs number of occurances
    4. Select the most frequent occuring adjacent token pair in current tokenized representation
    5. Merge that pair into a single merged token
    6. Replace every ocurance of that selected token pair in the current tokenied corpus
    7. Repeaat until the target vocabulary size is reached or no valid pair remains

"""
class BPETrainer:

    def __init__(self, corpus: list[str], target_vocab_size: int, min_pair_freq=1):
        # a list of text examples that is used to train the tokenizer, example: ["hello", "help", "helmet"]
        self.corpus = corpus.copy()
        # the desired final vocab size used for stopping condition
        self.target_vocab_size = target_vocab_size
        # the minimum number of occurrences required for a pair of tokens to be merged.
        self.min_pair_freq = min_pair_freq

        self.vocabulary = Vocabulary()                      # init the current vocabulary during the training of this tokenizer
        self.tokenized_corpus: TOKENIZED_CORPUS = []        # represents the corpus using the tokenizers current vocabulary, list of type TOKENIZED_CORPUS
        self.merge_rules: list[TokenPair] = []              # the rules of how token pairs are merged, list of token-pairs implies that each of these were merged to create a single token-pair

        self.is_initialized = False
        self.is_trained = False

    def train(self) -> BPETrainingResult:
        if self.is_trained:
            return self.create_training_result()
        
        # initially tokenize the corpus and add all unqiue characters in corpus as tokens to our vocab
        self.initialize_corpus_and_vocab()

        # set the initial-vocab-size which is length of vocab-size or its number of tokens
        initial_vocab_size = len(self.vocabulary)

        # check if target-vocab-size is smaller than init-vocab-size which is snot allowed
        if self.target_vocab_size < initial_vocab_size:
            raise ValueError("Target vocabulary size cannot be smaller than the " f"initial character vocabulary size. " f"Initial size: {initial_vocab_size}, " f"target size: {self.target_vocab_size}.")

        # while the current size of the vocabulary has not reached our target vocab size, keep running training iterations
        while len(self.vocabulary) < self.target_vocab_size:
            # get the adjacent-token-pairs occurances count for the current tokenized-corpus, dict = (token1, token2) -> frequency
            pair_counts_dict = self.count_adjacent_pairs()
            # get the most occuring adjacent-token-pair and its frequency in current tokenized corpus representation
            selected_pair, frequency = self.select_most_frequent_pair(pair_counts_dict)

            # if the frequency of the most occuring adajcent-token-pair is less than min-pair-freq, meaning that the most occuring adjacent-token-pair is not that common in the tokenized-corpus so its not worth merging, so stopping condition
            if frequency < self.min_pair_freq:
                break
            
            # merge that most occuring token-pair into a single token
            merged_token = self.merge_token_pair(selected_pair)

            # add that merged-token into our vocabulary as a new single token, it will be given an tokenID
            self.vocabulary.add_token(merged_token)
            # since we merged this token-pair it is a merge rule so simply add the token-pair as a merge rule keeping order
            self.merge_rules.append(selected_pair)
            # replace every occurance of the selected-token-pair in current-tokenized-corpus with the merged-token, and get the updated tokenized-corpus
            self.tokenized_corpus = self.tokenize_corpus_with_merged_token(selected_pair, merged_token)

        # are multiple iteratations of this and merging tokens and adding it to the vocabulary, the tokenizer has been trained
        self.is_trained = True
        # return everything we need after training vocabulary, merge rules, tokenized corpus
        return self.create_training_result()
            

    
    """
    Initialy tokenize the corpus and create the vocabulary-class with all characters as tokens. 

    Each text sample is initially represented using characteres cause char-level bpe.  Initialize the corpus representation and vocabulary. 
    Example:
        ["hello", "help"]
    becomes:
        [
            ["h", "e", "l", "l", "o"],
            ["h", "e", "l", "p"],
        ]
    """
    def initialize_corpus_and_vocab(self):
        if self.is_initialized:
            return
        
        # we need to tokenize the corpus but this is before the first iteration, and for char-level bpe all tokens are characters
        self.tokenized_corpus = []

        # for every text-sample-str in corpus
        for text in self.corpus:
            # tokenize the current text sample into list of characters because since right now the tokens are all characters, ["h", "e", "l", "p"]
            tokens = list(text)

            if not tokens:
                continue
            
            # add that list of characters of the text-sample or tokenized-text-sample to our tokenized-corpus, we are basically tokenizing the corpus using charcters as our vocab
            self.tokenized_corpus.append(tokens)

            # iterate through every token/char in cur-text-sample char-list, add that token to our tokenizer's vocab-list, so basically adding every unqiue character token to the vocabulary
            for cur_token in tokens:
                self.vocabulary.add_token(cur_token)    # cur_token = "a"
        
        self.is_initialized = True

    """
    Count the number of occurances of every adjacent token pair in the tokenized corpus representation
    """
    def count_adjacent_pairs(self) -> dict[tuple[str, str], int]:
        # count the occurances of every adjacent token pair in the current tokenized corpus representation, return a dict: (token1, token2) -> frequency
        pair_counts_dict: dict[tuple[str, str], int] = {}

        # iterate through every tokenized text sampe in corpus
        for token_sequence in self.tokenized_corpus:
            # iterate through every token in cur-token-sequence just before last one
            for i in range(len(token_sequence) - 1):
                # define the current adjacent token pair for this current token-i for this current token-sequence, i+1 token is the adjacent token 
                pair = (token_sequence[i], token_sequence[i+1])

                # if this adjacent-token-pair is not in the counts-dict, but its count to one, else we have seen this adjacent-token-pair before so increment its occurances-value
                if pair not in pair_counts_dict:
                    pair_counts_dict[pair] = 1
                else:
                    pair_counts_dict[pair] += 1

        return pair_counts_dict
    
    """
    Select the token pair that occurs the most times in current tokenized corpus given the dict that has (token1, token2) -> frequency
    """
    def select_most_frequent_pair(self, pair_counts_dict: dict[tuple[str, str], int]) -> tuple[TokenPair, int]:
        most_frequent_pair = max(pair_counts_dict, key=pair_counts_dict.get)    # get the most frequent adjcent token pair (token1, token2)
        frequency = pair_counts_dict[most_frequent_pair]                        # get the occurances of that most freq token pair

        return most_frequent_pair, frequency

    """
    Merge a token pair (t1, t2) into a single merged token t1t2
    ("h", "e") -> "he"
    ("he", "l") -> "hel"
    """
    def merge_token_pair(self, pair: TokenPair) -> str:
        left_token, right_token = pair

        return left_token + right_token
    
    """
    Replace every occurrence of selected adjacent pair in corpus current tokenized representation with the new merged token
    Note that:
        - old tokens remain in the vocabulary
        - selected occurrences are replaced in the tokenized corpus
        - the trainer does not retokenize from the vocabulary fmor scratch after every merge
        - it just applies exactly the newly selected merge to the current representation
        - this is because even though the vocabulary says all of these tokens exist, but it does not specify which representation should be used. The ordered merge rules determinet hat. 
    Pair:
        ("h", "e")

        Before:
            ["h", "e", "l", "p"]

        After:
            ["he", "l", "p"]
    """
    def tokenize_corpus_with_merged_token(self, pair: TokenPair, merged_token: str):    # ass in the token-pair and the merged-token
        updated_corpus: TOKENIZED_CORPUS = []

        # iterate every tokenized-text-sample in corpus, may be ["h", "e", "l", "p"]
        for token_sequence in self.tokenized_corpus:
            # store the updated-tokenized-sequence, may be ["he", "l", "p"]
            updated_sequence: list[str] = []
            # store the current index of the current tokenized sequence
            indx = 0

            # while index is still valid for token-sequence
            while indx < len(token_sequence):
                # find if current pair can be merged if indx there is stil room to test a pair AND the current-indx-token is equal to first token in pair AND the next-index-token is equal to second token in pair, then we have found the pair of tokens that need to be merged according to our new merge rule and new merge token
                pair_can_be_merged = (indx < len(token_sequence)-1 and token_sequence[indx] == pair[0] and token_sequence[indx + 1] == pair[1])
                
                # if cur-token-pair can be merged simply add the given merged-token to the cur-updated-tokenized-sequence, because the concatenation of t1+t2 is the merged-token
                if pair_can_be_merged:
                    updated_sequence.append(merged_token)
                    indx += 2      # update the index to go to the next token in sequence where we can check
                else:   # else if cur-token-pair cannot be merged meaning the two tokens are cannotnot matching then simply add the cur-indx-token in token-sequence to our updated-token-sequence
                    updated_sequence.append(token_sequence[indx])
                    indx += 1
                
            updated_corpus.append(updated_sequence) # add the re-tokenized sequence to our new corpus

        return updated_corpus   # return the new corpus 



    def create_training_result(self) -> BPETrainingResult:
        # create snapshot of the current training result output and return it
        return BPETrainingResult(
            vocabulary=self.vocabulary,
            merge_rules=self.merge_rules.copy(),
            tokenized_corpus=[
                sequence.copy() for sequence in self.tokenized_corpus
            ]
        )

# run: python -m tokenization.bpe_trainer, library/
if __name__ == "__main__":

    print("------Testing BPE Trainer------\n")

    corpus = [
        "hello",
        "help",
        "helmet",
    ]

    trainer = BPETrainer(
        corpus=corpus,
        target_vocab_size=10,
    )

    result = trainer.train()

    print("Vocabulary:")
    print(result.vocabulary.token_to_id)

    print("\nMerge Rules:")
    print(result.merge_rules)

    print("\nFinal Tokenized Corpus:")
    for text_sample in result.tokenized_corpus:
        print(text_sample)