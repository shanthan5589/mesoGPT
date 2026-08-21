import time

import pyarrow.parquet as pq
import requests

import argparse

from mesoGPT.common import ROOT_DIR

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

# Nanochat uses this fixed final shard for validation.
VALIDATION_SHARD_INDEX = 6542

DATA_DIRECTORY = ROOT_DIR / "data"

def shard_filename(index):
    """Convert a shard index into its filename."""
    return f"shard_{index:05d}.parquet"


def download_shard(index, max_attempts=5):
    """Download one Parquet shard safely with retries."""
    if not 0 <= index <= VALIDATION_SHARD_INDEX:
        raise ValueError(
            f"Shard index must be between 0 and "
            f"{VALIDATION_SHARD_INDEX}"
        )

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    filename = shard_filename(index)
    destination = DATA_DIRECTORY / filename
    temporary_file = DATA_DIRECTORY / f"{filename}.tmp"

    if destination.exists():
        print(f"Skipping {filename}: already downloaded")
        return True

    url = f"{BASE_URL}/{filename}"

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"Downloading {filename} "
                f"(attempt {attempt}/{max_attempts})..."
            )

            with requests.get(
                url,
                stream=True,
                timeout=30,
            ) as response:
                response.raise_for_status()

                with temporary_file.open("wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)

            # The final filename appears only after a successful download.
            temporary_file.replace(destination)

            print(f"Downloaded {filename}")
            return True

        except (requests.RequestException, OSError) as error:
            print(f"Download failed: {error}")

            # Remove only the incomplete temporary file.
            temporary_file.unlink(missing_ok=True)

            if attempt < max_attempts:
                wait_seconds = 2 ** attempt
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    print(f"Could not download {filename}")
    return False

def list_parquet_files():
    """Return all local Parquet shards in filename order."""
    parquet_files = sorted(DATA_DIRECTORY.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {DATA_DIRECTORY}"
        )

    return parquet_files


def parquet_batches(split):
    """
    Yield one batch of text documents at a time.

    Convention:
    - All shards except the last are training data.
    - The final shard is validation data.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")

    parquet_files = list_parquet_files()

    if len(parquet_files) < 2:
        raise ValueError(
            "At least two Parquet files are required: "
            "one for training and one for validation."
        )

    if split == "train":
        selected_files = parquet_files[:-1]
    else:
        selected_files = parquet_files[-1:]

    for parquet_path in selected_files:
        parquet_file = pq.ParquetFile(parquet_path)

        for row_group_index in range(parquet_file.num_row_groups):

            row_group = parquet_file.read_row_group(
                row_group_index,
                columns=["text"],
            )

            documents = row_group.column("text").to_pylist()

            yield documents

def main():
    parser = argparse.ArgumentParser(
        description="Download mesoGPT tokenizer data"
    )

    parser.add_argument(
        "-n",
        "--num-train-shards",
        type=int,
        default=1,
        help="Number of training shards to download",
    )

    args = parser.parse_args()

    if not 1 <= args.num_train_shards <= VALIDATION_SHARD_INDEX:
        raise ValueError(
            f"num-train-shards must be between 1 and "
            f"{VALIDATION_SHARD_INDEX}"
        )

    # Training shards begin at zero.
    shard_indices = list(range(args.num_train_shards))

    # Always include nanochat's fixed validation shard.
    shard_indices.append(VALIDATION_SHARD_INDEX)

    print(
        f"Downloading {args.num_train_shards} training shard(s) "
        "and one validation shard"
    )

    successful_downloads = 0

    for shard_index in shard_indices:
        if download_shard(shard_index):
            successful_downloads += 1

    print(
        f"Downloaded {successful_downloads}/"
        f"{len(shard_indices)} shards"
    )

    if successful_downloads != len(shard_indices):
        raise RuntimeError("One or more shard downloads failed")


if __name__ == "__main__":
    main()