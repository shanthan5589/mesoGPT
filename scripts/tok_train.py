from time import perf_counter

from mesoGPT.dataset import parquet_batches
from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import ROOT_DIR

VOCAB_SIZE = 4096
MAX_TRAINING_CHARACTERS = 10_000_000
MAX_CHARACTERS_PER_DOCUMENT = 10_000

TOKENIZER_OUTPUT_DIRECTORY = (
    ROOT_DIR / "artifacts" / "tokenizer"
)


def training_text_iterator():
    """Yield cropped documents for tokenizer training."""
    total_characters = 0

    for document_batch in parquet_batches(split="train"):

        for document in document_batch:
            if not document:
                continue

            remaining_characters = (
                MAX_TRAINING_CHARACTERS - total_characters
            )

            if remaining_characters <= 0:
                return
            elif remaining_characters <= MAX_CHARACTERS_PER_DOCUMENT:
                document = document[:remaining_characters]
            elif remaining_characters > MAX_CHARACTERS_PER_DOCUMENT:
                document = document[:MAX_CHARACTERS_PER_DOCUMENT]

            total_characters += len(document)

            yield document

            if total_characters >= MAX_TRAINING_CHARACTERS:
                return


def main():
    
    print("Training mesoGPT tokenizer...")
    print(f"Vocabulary size: {VOCAB_SIZE:,}")
    print(
        f"Maximum training characters: "
        f"{MAX_TRAINING_CHARACTERS:,}"
    )

    start_time = perf_counter()

    tokenizer = BPETokenizer.train_from_iterator(
        text_iterator=training_text_iterator(),
        vocab_size=VOCAB_SIZE,
    )

    elapsed_time = perf_counter() - start_time

    print(f"Training completed in {elapsed_time:.2f} seconds")

    tokenizer.save(TOKENIZER_OUTPUT_DIRECTORY)

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
            f"Original: {test_text!r}\n"
            f"Decoded:  {decoded_text!r}"
        )

    print("Round-trip test passed")
    print(f"Test characters: {len(test_text)}")
    print(f"Test tokens: {len(token_ids)}")
    print(f"BOS token ID: {tokenizer.get_bos_token_id()}")
    print(f"Saved to: {TOKENIZER_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()