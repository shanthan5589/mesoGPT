import pyarrow.parquet as pq
from mesoGPT.dataset import list_parquet_files
import argparse
import math

def main(type, max_training_chars, max_characters_per_document=0):

    if type == "tokenizer":
        #raw_characters = 0
        eligible_characters = 0
        available_shards = len(list_parquet_files()) - 2

        if available_shards <= 2:
            print("No training shards found. Please run dataset.py to create download shards.")
            exit()

        # Last shard is reserved for validation, so we only count training shards.
        for parquet_path in list_parquet_files()[:-2]:
            
            parquet_file = pq.ParquetFile(parquet_path)

            #document_count = 0

            for record_batch in parquet_file.iter_batches(
                columns=["text"]
            ):
                documents = record_batch.column("text").to_pylist()

                for document in documents:
                    if not document:
                        continue

                    #document_count += 1
                    #raw_characters += len(document)

                    # This matches tok_train.py's document cropping.
                    eligible_characters += len(
                        document[:max_characters_per_document]
                    )

            #print(f"{parquet_path.name}  Documents: {document_count:,}")

        #print(f"  Raw characters: {raw_characters:,}")
        print(f"  Eligible characters available to train tokenizer: {eligible_characters:,}")

        required_extra_shards = math.ceil((max_training_chars - eligible_characters) / (eligible_characters / available_shards)) if eligible_characters < max_training_chars else 0 

        return available_shards, required_extra_shards, eligible_characters >= max_training_chars
    elif type == "model":

        raw_characters = 0
        available_shards = len(list_parquet_files()) - 2

        if available_shards <= 0:
            print("No training shards found. Please run dataset.py to create download shards.")
            exit()

        # Last shard is reserved for validation, so we only count training shards.
        for parquet_path in list_parquet_files()[:-2]:
            
            parquet_file = pq.ParquetFile(parquet_path)

            #document_count = 0

            for record_batch in parquet_file.iter_batches(
                columns=["text"]
            ):
                documents = record_batch.column("text").to_pylist()

                for document in documents:
                    if not document:
                        continue

                    #document_count += 1
                    raw_characters += len(document)

                    # # This matches tok_train.py's document cropping.
                    # eligible_characters += len(
                    #     document[:max_characters_per_document]
                    # )

            #print(f"{parquet_path.name}  Documents: {document_count:,}")

        #print(f"  Raw characters: {raw_characters:,}")
        print(f"  Raw characters available to train tokenizer: {raw_characters:,}")

        required_extra_shards = math.ceil((max_training_chars - raw_characters) / (raw_characters / available_shards)) if raw_characters < max_training_chars else 0 

        return available_shards, required_extra_shards, raw_characters >= max_training_chars

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
        "--max-training-chars",
        type=int,
        default=10**9,
        help="Maximum number of characters to train tokenizer on.",
    )

    parser.add_argument(
        "--max-chars-per-document",
        type=int,
        default=10**3,
        help="Maximum number of characters to use from each document.",
    )

    args = parser.parse_args()

    available_shards, required_extra_shards, enough_characters = main(
        args.type, args.max_training_chars, args.max_chars_per_document
    )

    if not enough_characters:
        print(f"  {available_shards} shards available, you need more {required_extra_shards:.2f} shards to train the {args.type}.")
    else:
        print(f"  {available_shards} shards available, you have enough characters to train the {args.type}.")