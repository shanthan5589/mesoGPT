import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mesoGPT.dataset import parquet_batches

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

                    end = start + self.context_length

                    xb, yb = token_ids[start:end], token_ids[start + 1:end + 1]

                    yb_length = len(yb)

                    num_bytes = len(self.tokenizer.decode(yb).encode("utf-8"))

                    xb = torch.tensor(xb, dtype=torch.long)
                    yb = torch.tensor(yb, dtype=torch.long)

                    yield xb, yb, num_bytes, yb_length

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