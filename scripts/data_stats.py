'''
available arguments:
    type: "tokenizer" or "model"
    --max-budget: maximum number of characters/tokens used to train tokenizer/model.
    --max-chars-per-document: maximum number of characters to use from each selected document (only applicable when type is "tokenizer")
'''


from mesoGPT.dataset import list_parquet_files, parquet_batches
from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import TOKENIZER_DIR, TOKENIZER_NAME

import argparse
import math

def main(type, max_budget, max_characters_per_document=0):

    if type not in {"tokenizer", "model"}:
        raise ValueError("type must be either 'tokenizer' or 'model'")

    available_training_shards = len(list_parquet_files()) - 2

    if available_training_shards < 1:
        print("At least 3 shards are required to train the tokenizer or model. Please download more shards.")
        exit()

    if type == "tokenizer":

        eligible_characters_per_shard = 0

        for documents in parquet_batches('one'):

            for document in documents:

                if not document:
                    continue

                eligible_characters_per_shard += len(document[:max_characters_per_document])

        required_extra_shards = math.ceil((max_budget - (eligible_characters_per_shard * available_training_shards)) / eligible_characters_per_shard) + 1 if eligible_characters_per_shard < max_budget else 0 

        if required_extra_shards > 0:
            storage_needed = ((available_training_shards + required_extra_shards) * 92) / 1024

        return available_training_shards, required_extra_shards, storage_needed if required_extra_shards > 0 else 0
    
    if type == "model":

        tokenizer = BPETokenizer.from_directory(
            tokenizer_directory=TOKENIZER_DIR,
            tokenizer_name=TOKENIZER_NAME,
        )

        tokens_per_shard = 0
        available_training_shards = len(list_parquet_files()) - 2

        for documents in parquet_batches('one'):

            for document in documents:

                if not document:
                    continue

                tokens_per_shard += len(tokenizer.encode(document))

        print(f"Tokens per shard: {tokens_per_shard:,}")

        required_extra_shards = math.ceil((max_budget - (tokens_per_shard * available_training_shards)) / tokens_per_shard) + 1 if tokens_per_shard < max_budget else 0 

        storage_needed = ((available_training_shards + required_extra_shards) * 92) / 1024

        return available_training_shards, required_extra_shards, storage_needed

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Count the number of characters in the training data."
        )

    parser.add_argument(
        "type",
        choices=["tokenizer", "model"],
        help="Type of character count to perform: 'tokenizer' counts eligible characters for tokenizer training, 'model' counts raw characters for model training.",
    )
    
    parser.add_argument(
        "--max-budget",
        type=int,
        default=10**9,
        help="Maximum number of characters/tokens to train tokenizer/model on.",
    )

    parser.add_argument(
        "--max-chars-per-document",
        type=int,
        default=10**3,
        help="Maximum number of characters to use from each document.",
    )

    args = parser.parse_args()

    available_training_shards, required_extra_shards, storage_needed = main(
        args.type, args.max_budget, args.max_chars_per_document
    )

    if required_extra_shards > 0:
        print(f"  {available_training_shards} shards available, you need more {required_extra_shards:.2f} shards to train the {args.type}. Total storage needed: {storage_needed:.2f} GiB.")
    else:
        print(f"  {available_training_shards} shards available, you have enough characters to train the {args.type}.")