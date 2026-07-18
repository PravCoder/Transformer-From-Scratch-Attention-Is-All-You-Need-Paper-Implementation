"""
FILE: manages the vocabulary, only responsibility is managing bidirectional mappings between tokens and IDs

Token: is defined is a unit of text after tokenization. 
Token ID: is unqiue numerical integer assigned to token.
"""


"""
Stores the tokens known by a tokenizer and maintains both mappings:
token -> tokenID
tokenID -> token

Token IDs are assigned sequentially starting from 0. 

Note tokens are of datatype string and tokenIDs are of type int.
"""
class Vocabulary:

    def __init__(self):
        self.token_to_id: dict[str, int] = {}       # mapping from token-str to tokenID-int
        self.id_to_token: dict[int, str] = {}       # mapping from tokenID-int to token-str

    # adds a given token-str to the vocab which means adding it to both mappings
    def add_token(self, token: str) -> int:
        # if param is not str raise error
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, received {type(token).__name__}.")
        # token must non-empty string
        if token == "":
            raise ValueError("Token cannot be empty")
        # if token already exists in vocab, return its exsiting id
        if token in self.token_to_id:
            return self.token_to_id[token]

        # else token does not already exist, generate id for it, increment to next integer for id
        token_id = len(self.token_to_id)
        # add new token and tokenID to both mappings, adding it to our tokenizers vocab
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

        return token_id     # return the new token-id

    # given a token-str get its tokenID using our vocabulary
    def get_id(self, token: str) -> int:
        # if the token is not in the vocabulary
        if token not in self._token_to_id:
            raise KeyError(f"Token {token!r} is not in the vocabulary.")
        
        # use token-str to get tokenID
        return self.token_to_id[token]
    
    # given a token-ID get its token-str using our vocabulary
    def get_token(self, token_id: int) -> str:
        # if the tokenID is not in vocab
        if token_id not in self._id_to_token:
            raise KeyError(f"Token ID {token_id} is not in the vocabulary.")
        
        # use tokenID to get its token-str
        return self.id_to_token[token_id]
    

    # return t/f if token exists in vocab
    def contains_token(self, token: str) -> bool:
        return token in self.token_to_id
    
    # return t/f if tokenID exists in vocab
    def contains_id(self, token_id: int) -> bool:
        return token_id in self.id_to_token

    # returns a copy of token-to-ID mapping
    @property
    def _token_to_id(self) -> dict[str, int]:
        return self.token_to_id.copy()
    
    # returns a copy of ID-to-token mapping
    @property
    def _id_to_token(self) -> dict[str, int]:
        return self.id_to_token.copy()
    
    # supports the "hello" in vocabulary in python native, implement our own custom in method for this class
    def __contains__(self, token: str) -> bool:
        return self.contains_token(token)
    
    # return vocab size, number of known tokens
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    # print object
    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"


if __name__ == "__main__":
    print("\n------Testing Vocabulary of Tokenizer-----")
    vocabulary = Vocabulary()

    h_id = vocabulary.add_token("h")
    e_id = vocabulary.add_token("e")
    he_id = vocabulary.add_token("he")

    print(h_id)                 # 0
    print(e_id)                 # 1
    print(he_id)                # 2

    print(vocabulary.get_id("he"))  # 2
    print(vocabulary.get_token(2))  # "he"

    print(len(vocabulary))      # 3
    print("he" in vocabulary)   # True

    first_id = vocabulary.add_token("he")
    second_id = vocabulary.add_token("he")

    print(first_id == second_id)  # True
    print(len(vocabulary))        # Still 3