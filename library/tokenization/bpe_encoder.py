"""
FILE: implements the BPE Encoding Algorithm
"""

from tokenization.vocabulary import Vocabulary

# a typed variable to represent two paired tokens
TokenPair = tuple[str, str]


"""
Represents the tokenizers encoding. Encodes text into token IDs using the trained BPE vocabulary and the ordered merge rules learned duinrg BPE training
"""
class BPEEncoder:

    def __init__(self, vocabulary: Vocabulary, merge_rules: list[TokenPair]):
        self.vocabulary = vocabulary
        self.merge_rules = merge_rules.copy()

    """
    Converts raw text into BPE token IDs. Returns a list of integers represtning the tokenized text
    Process:
        1. Represent the text using the base-level tokens which are the initial character tokens
        2. Apply learned merge rules in order.
        3. Convert the resulting tokens into token IDs.
    """
    def encode(self, text: str) -> list[int]:
        # tokenize the raw text, meaning convert it into ba-selevel char tokens first, then apply the merge rules in order of those tokens 
        tokens = self.tokenize(text)

        token_ids = []

        # iterate all tokens after tokenization (base-tokens + merge-rules) and convert each token into its token-ID using our vocabulary, and return list of tokenIDs
        for cur_token in tokens:
            cur_token_id = self.vocabulary.get_id(cur_token)
            token_ids.append(cur_token_id)

        return token_ids


    """
    Convert raw text into its final BPE token representation without converting the tokens into IDs yet
    """
    def tokenize(self, text: str) -> list[str]:
        # first convert the given text into a list of characters because classic BPE starts new text as individual character tokens
        tokens = list(text)

        # for every token-pair in the merge-rules we learned during training, apply that merge-rule to the list of toekns
        for pair in self.merge_rules:
            # the merged-token is just the concat of the tokens in the cur-pair in merge-rules
            merged_token = pair[0] + pair[1]

            # apply this single merge rule to the current tokenized representation give the merged-token and the merge-rule or pair
            tokens = self.apply_merge_rule(tokens=tokens, pair=pair, merged_token=merged_token)

        return tokens

    """
    Apply a given merge-rule (pair) on the given list of tokens strings, and given the merged-token. 
    Repalce every non-overlapping occurrence of the token-pair with its merged token
    Example:
        tokens = ["h", "e", "l", "p"]
        pair = ("h", "e")
        merged_token = "he"

        result = ["he", "l", "p"]
    """
    def apply_merge_rule(self, tokens: list[str], pair: TokenPair, merged_token: str) -> list[str]:

        # the updated list of tokens after applying the given merge-rule
        updated_tokens: list[str] = []
        # the current index in the tokens
        indx = 0

        # while is the current index of the tokens hasnt reached the end, check each adjacent pair of tokens in the sequence
        while indx < len(tokens):  
            # the current adjacent token pair in tokens-sequence can be merged if the cur-index as room to check a pair AND the cur-index-token is equal to the first token in the merge-rule-pair AND the next-index-token is equal to the second token in the merge-rule-pair
            pair_can_be_merged = (indx < len(tokens) -1 and tokens[indx] == pair[0] and tokens[indx + 1] == pair[1])

            if pair_can_be_merged:
                updated_tokens.append(merged_token) # if pair can be merged just add the merged-token (the concat of the pair) as the cur-index-token to the updated tokens
                indx += 2                           # move the indx to the next avaible token to check
            else:
                updated_tokens.append(tokens[indx]) # if pair cannot be merged just add the cur-indx-token, no change needed
                indx += 1                           # noramlly increment to the next token

        return updated_tokens


# run: python -m tokenization.bpe_encoder, library/
if __name__ == "__main__":
    from bpe_trainer import BPETrainer

    print("------ Testing BPE Encoder ------\n")

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

    text = "hello"

    tokens = encoder.tokenize(text)
    token_ids = encoder.encode(text)

    print("Original text:")
    print(text)

    print("\nBPE tokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids)