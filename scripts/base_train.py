import torch
import torch.nn as nn

from mesoGPT.model_1 import GPT as GPT_1
from mesoGPT.model_2 import GPT as GPT_2
from mesoGPT.model_3 import GPT as GPT_3
from mesoGPT.model_4 import GPT as GPT_4


from mesoGPT.dataloader import create_dataloader

from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import TOKENIZER_DIR, TOKENIZER_NAME, CHECKPOINT_DIR, MODEL_NAME

import argparse

import math

'''
# ---------------- hyperparameters ----------------

# Model
context_length = 256
n_embed = 384
n_layers = 6
n_heads = 6

# Training:
max_steps = 5000
learning_rate = 3e-4
batch_size = 64
dropout = 0.2

# Evaluation
eval_interval = 250
eval_iters = 50

# Dataset
stride = context_length
drop_last = True        # Set to True for Validation dataset
num_workers = 0

# -------------------------------------------------
'''

max_lr = 3e-4       # 0.0003
min_lr = max_lr * 0.1  # 0.00003
warmup_steps = 10
max_steps = 50


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


@torch.no_grad()
def estimate_loss(model, tokenizer,criterion, 
                  eval_iters, batch_size, vocab_size, context_length, 
                  stride, drop_last, num_workers):

    model.eval()
    losses = {}

    train_dataloader_eval = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        repeat=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    val_dataloader = create_dataloader(
        split="model_val",
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        repeat=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    for split, dataloader in [
        ("train", train_dataloader_eval),
        ("val", val_dataloader),
    ]:

        batch_losses = []

        for batch_index, (xb, yb) in enumerate(dataloader):

            if batch_index >= eval_iters:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

            batch_losses.append(loss.item())
            
        losses[split] = sum(batch_losses) / len(batch_losses)

    model.train()
    return losses


def getlr(steps):
    if steps < warmup_steps:
        progress = (steps + 1) / warmup_steps
        learning_rate = max_lr * progress
        return learning_rate

    if steps > max_steps:
        learning_rate = min_lr
        return learning_rate
    
    decay_ratio = (steps - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

def train(model, tokenizer, optimizer, criterion, 
          max_steps, eval_interval, eval_iters, 
          batch_size, vocab_size, context_length, 
          stride, drop_last, num_workers):

    assert eval_interval > 0, "eval_interval must be greater than 0 to avoid division by zero error."
    
    model.train()
    train_dataloader = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        repeat=True,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    train_iterator = iter(train_dataloader)

    best_val_loss = float("inf")

    for step in range(max_steps):

        lr = getlr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = next(train_iterator)

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)      
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()

        optimizer.step()

        completed_steps = step + 1

        if completed_steps % eval_interval == 0 or completed_steps == max_steps:

            losses = estimate_loss(
                model=model,
                tokenizer=tokenizer,
                criterion=criterion,
                eval_iters=eval_iters,
                batch_size=batch_size,
                vocab_size=vocab_size,
                context_length=context_length,
                stride=stride,
                drop_last=drop_last,
                num_workers=num_workers
            )

            print(f"Step: {completed_steps}: "  
                f"Train Loss: {losses['train']:.4f}, "
                f"Val Loss: {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss:

                best_val_loss = losses['val']

                torch.save({
                    "model_args": {
                        "T": args.context_length,
                        "C": args.n_embed,
                        "vocab_size": vocab_size,
                        "num_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout
                    },
                    "state_dict": model.state_dict(),
                    # Useful if you want to resume training
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": completed_steps,
                    "val_loss": best_val_loss,
                }, CHECKPOINT_DIR / f"{MODEL_NAME}_{args.model_version}.pt")

                print(
                        f"Saved new best checkpoint "
                        f"with validation loss {best_val_loss:.4f}"
                    )


if __name__ == "__main__":

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument("--model_version", type=int, default=1, choices=[1, 2, 3, 4], help="Version of the GPT model to use.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model.")
    parser.add_argument("--n_embed", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate.")
    parser.add_argument("--eval_interval", type=int, default=250, help="Interval for evaluation during training.")
    parser.add_argument("--eval_iters", type=int, default=50, help="Number of iterations for evaluation.")
    parser.add_argument("--stride", type=int, default=None, help="Stride for the dataset. Defaults to context_length if not provided.")
    parser.add_argument("--drop_last", action='store_true', help="Whether to drop the last incomplete batch in the dataloader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for the dataloader.")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_directory(
        tokenizer_directory=TOKENIZER_DIR,
        tokenizer_name=TOKENIZER_NAME,
    )

    vocab_size = tokenizer.get_vocab_size()

    model_classes = {
        1: GPT_1,
        2: GPT_2,
        3: GPT_3,
        4: GPT_4,
    }

    model_class = model_classes[args.model_version]

    model = model_class(vocab_size=vocab_size, 
                T=args.context_length, 
                C=args.n_embed, 
                n_layers=args.n_layers, 
                num_heads=args.n_heads, 
                dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        vocab_size=vocab_size,
        context_length=args.context_length,
        stride=args.stride if args.stride is not None else args.context_length,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        args=args
    )