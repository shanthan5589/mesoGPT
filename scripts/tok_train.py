from time import perf_counter

from mesoGPT.dataset import list_parquet_files, parquet_batches
from mesoGPT.tokenizer import BPETokenizer
import pyarrow.parquet as pq
from mesoGPT.common import ROOT_DIR

import argparse

from pathlib import Path

import re

TOKENIZER_OUTPUT_DIRECTORY = (
    ROOT_DIR / "artifacts" / "tokenizer"
)


def training_text_iterator(args):
    """Yield cropped documents for tokenizer training."""
    total_characters = 0

    for document_batch in parquet_batches(split="train"):

        for document in document_batch:
            if not document:
                continue

            remaining_characters = (
                args.max_training_chars - total_characters
            )

            if remaining_characters <= 0:
                return
            elif remaining_characters <= args.max_chars_per_document:
                document = document[:remaining_characters]
            elif remaining_characters > args.max_chars_per_document:
                document = document[:args.max_chars_per_document]

            total_characters += len(document)

            yield document

            if total_characters >= args.max_training_chars:
                return

def validation(tokenizer):

    total_characters = 0
    total_bytes = 0
    total_tokens = 0
    word_count = 0

    if len(list_parquet_files()) <= 2:
        print("No validation shards found. Please run dataset.py to create download shards.")
        exit()
    
    # Last but one shard is reserved for validation of tokenizer, so we only count the validation shard.
    for document_batch in parquet_batches(split="tokenizer_val"):
                
        for document in document_batch:

            if not document:
                continue

            token_ids = tokenizer.encode(document)

            total_characters += len(document)
            total_bytes += len(document.encode("utf-8"))
            total_tokens += len(token_ids)
            words = re.findall(r"\b\w+\b", document)
            word_count += len(words)

    characters_per_token = (total_characters / total_tokens)
    bytes_per_token = total_bytes / total_tokens
    tokens_per_word = total_tokens / word_count

    return characters_per_token, bytes_per_token, tokens_per_word, total_characters, total_bytes, total_tokens, word_count



def main():

    parser = argparse.ArgumentParser(
            description="Train a BPE tokenizer on the training data."
        )
    
    parser.add_argument(
            "--max-training-chars",
            type=int,
            default=10**9,
            help="Maximum number of characters to train tokenizer on.",
        )
    
    parser.add_argument(
            "--max-chars-per-document",
            type=int,
            default=10**4,
            help="Maximum number of characters to use from each document.",
        )

    parser.add_argument(
            "--vocab-size",
            type=int,
            default=1000,
            help="Vocabulary size for the tokenizer.",
        )

    parser.add_argument(
        "--tokenizer-output-directory",
        type=Path,
        default=TOKENIZER_OUTPUT_DIRECTORY,
        help="Parent directory for trained tokenizers.",
    )
    
    args = parser.parse_args()

    training_char_millions = args.max_training_chars / 1_000_000

    tokenizer_name = (f"tok-v{args.vocab_size}" f"-c{training_char_millions:.2f}m")

    print("Training mesoGPT tokenizer...")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(
        f"Maximum training characters: "
        f"{args.max_training_chars:,}"
    )

    start_time = perf_counter()

    tokenizer = BPETokenizer.train_from_iterator(
        text_iterator=training_text_iterator(args),
        vocab_size= args.vocab_size,
    )

    elapsed_time = perf_counter() - start_time

    print(f"Training completed in {elapsed_time:.2f} seconds")

    characters_per_token, bytes_per_token, tokens_per_word, total_characters, total_bytes, total_tokens, word_count = validation(tokenizer)
    print(f"Validation results:")
    print(f"  Characters per token: {characters_per_token:.2f}")
    print(f"  Bytes per token: {bytes_per_token:.2f}")
    print(f"  Tokens per word: {tokens_per_word:.2f}")
    print(f"  Total characters: {total_characters:,}")
    print(f"  Total bytes: {total_bytes:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total words: {word_count:,}")
    print(f"BOS token ID: {tokenizer.get_bos_token_id()}")

    # Verify that encoding and decoding preserve the original text.
    test_text = """Hello from mesoGPT!
    Numbers: 12345
    Unicode: café, 你好, 🌍
    Contractions: I'm, you're, we'll"""

    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)

    if decoded_text != test_text:
        raise RuntimeError(
            "Tokenizer round-trip test failed:\n"
            # f"Original: {test_text!r}\n"
            # f"Decoded:  {decoded_text!r}"
        )

    print("Round-trip test passed")

    tokenizer.save(args.tokenizer_output_directory, tokenizer_name)

    print("="*50,end="")
    print()

if __name__ == "__main__":
    main()