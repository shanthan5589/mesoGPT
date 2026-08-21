import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, grammar):
        self.itos = {i:s for i,s in enumerate(grammar)}
        self.stoi = {s:i for i,s in enumerate(grammar)}

    def encode(self, content):
        return [self.stoi[x] for x in content]

    def decode(self, content):
        return ''.join([self.itos[x] for x in content])


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, T, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - T, stride):
            self.input_ids.append(torch.tensor(token_ids[i: i+T]))
            self.output_ids.append(torch.tensor(token_ids[i+1: i+T+1]))

        self.input_ids = torch.stack(self.input_ids)
        self.output_ids = torch.stack(self.output_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]

    def __len__(self):
        return len(self.input_ids)

    def split_data(self, train_ratio=0.8):
        split_idx = int(len(self.input_ids) * train_ratio)
        train_input_ids = self.input_ids[:split_idx]
        train_output_ids = self.output_ids[:split_idx]
        val_input_ids = self.input_ids[split_idx:]
        val_output_ids = self.output_ids[split_idx:]

        return (train_input_ids, train_output_ids), (val_input_ids, val_output_ids)

class TrainDataset(Dataset):
    def __init__(self, data):
        self.train_input_ids, self.train_output_ids = data

    def __getitem__(self, index):
        return self.train_input_ids[index], self.train_output_ids[index]

    def __len__(self):
        return len(self.train_input_ids)

class ValDataset(Dataset):
    def __init__(self, data):
        self.val_input_ids, self.val_output_ids = data

    def __getitem__(self, index):
        return self.val_input_ids[index], self.val_output_ids[index]

    def __len__(self):
        return len(self.val_input_ids)

def GPTDataLoader(dataset, B=4, shuffle=True, drop_last=True, num_workers=0):

    dataloader = DataLoader(dataset, batch_size=B, 
                            shuffle=shuffle, num_workers=num_workers, 
                            drop_last=drop_last)

    return dataloader


def load_data(B, T, stride, shuffle, drop_last, num_workers, train_ratio=0.8):

    with open('data/input.txt') as f:
        text = f.read()

    vocab = sorted(list(set(text)))

    tokenizer = Tokenizer(vocab)

    gpt_dataset = GPTDataset(text, tokenizer, T=T, stride=stride)

    train_dataset, val_dataset = gpt_dataset.split_data(train_ratio=train_ratio)

    train_dataloader = GPTDataLoader(TrainDataset(train_dataset), B=B, 
                                     shuffle=shuffle, 
                                     drop_last=drop_last, 
                                     num_workers=num_workers)
    
    val_dataloader = GPTDataLoader(ValDataset(val_dataset), B=B, 
                                     shuffle=False,
                                     drop_last=True, 
                                     num_workers=num_workers)

    return vocab, tokenizer, train_dataloader, val_dataloader