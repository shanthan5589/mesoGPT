import pickle
from pathlib import Path

import rustbpe
import tiktoken

# These tokens are reserved now so that we can support chat training later
# without retraining the tokenizer.
SPECIAL_TOKENS = [

    # Beginning Of Sequence.
    "<|bos|>",

    "<|user_start|>",
    "<|user_end|>",

    "<|assistant_start|>",
    "<|assistant_end|>",

    "<|python_start|>",
    "<|python_end|>", 

    "<|output_start|>",
    "<|output_end|>"
]

SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

class BPETokenizer:

    def __init__(self, encoding):
        self.tokenizer = encoding

        # Look up and remember the integer ID assigned to <|bos|>.
        self.bos_token_id = self.tokenizer.encode_single_token("<|bos|>")

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_special_tokens(self):
        return self.tokenizer.special_tokens_set

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # Special tokens occupy part of the requested vocabulary.
        bpe_vocab_size = vocab_size - len(SPECIAL_TOKENS)

        if bpe_vocab_size < 256:
            raise ValueError(
                "Vocabulary must have room for all 256 byte values "
                "and the special tokens."
            )

        # Step 1: use rustbpe to learn the vocabulary.
        trainer = rustbpe.Tokenizer()

        trainer.train_from_iterator(
            text_iterator,
            bpe_vocab_size,
            pattern=SPLIT_PATTERN,
        )

        # Step 2: retrieve the learned byte tokens and their ranks.
        learned_ranks = trainer.get_mergeable_ranks()

        mergeable_ranks = {
            bytes(token_bytes): token_id
            for token_bytes, token_id in learned_ranks
        }

        # Step 3: place special tokens after the learned BPE tokens.
        first_special_id = len(mergeable_ranks)

        special_tokens = {
            token: first_special_id + index
            for index, token in enumerate(SPECIAL_TOKENS)
        }

        # Step 4: create the fast tiktoken encoder.
        encoding = tiktoken.Encoding(
            name="mesogpt",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        return cls(encoding)

    def encode_special(self, token):
        """Convert a reserved special token into its integer ID."""
        return self.tokenizer.encode_single_token(token)

    def encode(self, text, prepend=None, append=None):
        """Convert ordinary text into token IDs."""
        token_ids = self.tokenizer.encode_ordinary(text)

        if prepend is not None:
            prepend_id = (
                prepend
                if isinstance(prepend, int)
                else self.encode_special(prepend)
            )
            token_ids.insert(0, prepend_id)

        if append is not None:
            append_id = (
                append
                if isinstance(append, int)
                else self.encode_special(append)
            )
            token_ids.append(append_id)

        return token_ids

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return self.tokenizer.decode(token_ids)

    def save(self, tokenizer_directory, tokenizer_name):
        """Save the trained tokenizer to a directory."""
        tokenizer_directory = Path(tokenizer_directory)
        tokenizer_directory.mkdir(parents=True, exist_ok=True)

        tokenizer_path = tokenizer_directory / f"{tokenizer_name}.pkl"

        with tokenizer_path.open("wb") as file:
            pickle.dump(self.tokenizer, file)

        print(f"Saved tokenizer to {tokenizer_path}")

    @classmethod
    def from_directory(cls, tokenizer_directory, tokenizer_name):
        """Load a previously trained tokenizer."""
        tokenizer_path = Path(tokenizer_directory) / f"{tokenizer_name}.pkl"

        with tokenizer_path.open("rb") as file:
            encoding = pickle.load(file)

        return cls(encoding)