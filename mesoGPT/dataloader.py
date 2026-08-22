import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mesoGPT.dataset import parquet_batches


class Tokenizer:
    def __init__(self):

        grammar = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!
        ?;:'\"()[]{}<>@#$%^&*-_=+|/\\`~\n\t"""

        self.itos = {i:s for i,s in enumerate(grammar)}
        self.stoi = {s:i for i,s in enumerate(grammar)}

    def encode(self, content):
        unknown_id = self.stoi[" "]
        return [self.stoi.get(character, unknown_id) for character in content]

    def decode(self, content):
        return ''.join([self.itos[x] for x in content])

    def __len__(self):
        return len(self.itos)


class ParquetTokenDataset(IterableDataset):
    def __init__(self, split, tokenizer, context_length, stride, repeat=False):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.repeat = repeat

    def __iter__(self):

        worker = get_worker_info()

        while True:
            for batch_index, document_batch in enumerate(
                parquet_batches(self.split)
            ):
                # Prevent multiple DataLoader workers from yielding
                # the same row group.
                if (
                    worker is not None
                    and batch_index % worker.num_workers != worker.id
                ):
                    continue

                documents = [
                    document
                    for document in document_batch
                    if document
                ]

                text = "\n".join(documents)
                token_ids = self.tokenizer.encode(text)

                for start in range(0, len(token_ids) - self.context_length, self.stride):

                    end = start + self.context_length + 1

                    window = torch.tensor(token_ids[start:end], dtype=torch.long)

                    yield window[:-1], window[1:]

            if not self.repeat:
                return


def create_dataloader(
    split,
    tokenizer,
    context_length,
    stride,
    batch_size,
    repeat=False,
    drop_last=True,
    num_workers=0,
):
    dataset = ParquetTokenDataset(
        split=split,
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        repeat=repeat,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
    )