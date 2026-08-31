import torch
import torch.nn as nn

import argparse
import math

from pathlib import Path

import json
from dataclasses import asdict

from model import GPT, GPTConfig

from mesoGPT.dataloader import create_dataloader
from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import TOKENIZER_DIR, TOKENIZER_NAME



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
        drop_last=False,
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


        if not batch_losses:
            raise RuntimeError(
                f"No evaluation batches were produced for split {split!r}."
            )
            
        losses[split] = sum(batch_losses) / len(batch_losses)

    model.train()
    return losses


def getlr(step, max_steps, warmup_steps, max_lr, min_lr):

    if step < warmup_steps:
        progress = (step + 1) / warmup_steps
        learning_rate = max_lr * progress
        return learning_rate

    if step >= max_steps:
        learning_rate = min_lr
        return learning_rate
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

def train(model, tokenizer, optimizer, criterion, 
          optimizer_steps, eval_interval, eval_iters, 
          batch_size, vocab_size, context_length, 
          stride, drop_last, num_workers, run_dir, args):

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

    for step in range(optimizer_steps):

        lr = getlr(step, args.max_lr_schedule_steps, args.warmup_steps, args.max_lr, args.min_lr)
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

        if completed_steps % eval_interval == 0 or completed_steps == optimizer_steps:

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

                model_path = run_dir / "model.pt"
                optimizer_path = run_dir / "model_optimizer.pt"
                metadata_path = run_dir / "model_meta.json"

                # Model weights only.
                torch.save(model.state_dict(), model_path)

                # Optimizer state kept separately for resuming training.
                torch.save(optimizer.state_dict(), optimizer_path)

                # Human-readable model and training metadata.
                metadata = {
                    "checkpoint_format_version": 1,
                    "run_id": run_dir.name,
                    "exp_no": "002",
                    "model_config": asdict(model.config),
                    "training_config": vars(args).copy(),
                    "tokenizer_name": TOKENIZER_NAME,
                    "step": completed_steps,
                    "val_loss": best_val_loss,
                }

                with metadata_path.open("w", encoding="utf-8") as file:
                    json.dump(metadata, file, indent=2)

                print(
                        f"Saved new best checkpoint "
                        f"with validation loss {best_val_loss:.4f}"
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GPT model.")

    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model.")
    parser.add_argument("--n_embed", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, required=False, help="Learning rate for the optimizer.")
    parser.add_argument("--optimizer_steps", type=int, default=5000, help="Maximum number of training steps.")

    parser.add_argument("--max_lr", type=float, default=3e-4, required=False, help="Maximum learning rate for the optimizer.")
    parser.add_argument("--min_lr", type=float, default=3e-5, required=False, help="Minimum learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=10, required=False, help="Number of warmup steps for learning rate scheduling.")
    parser.add_argument("--max_lr_schedule_steps", type=int, default=50, required=False, help="Total number of steps for learning rate scheduling.")

    parser.add_argument("--eval_interval", type=int, default=250, help="Interval for evaluation during training.")
    parser.add_argument("--eval_iters", type=int, default=50, help="Number of iterations for evaluation.")

    parser.add_argument("--stride", type=int, default=None, help="Stride for the dataset. Defaults to context_length if not provided.")
    parser.add_argument("--drop_last", action="store_true", help="Whether to drop the last incomplete batch in the dataloader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for the dataloader.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory where all artifacts for this run are saved.",
)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.from_directory(
        tokenizer_directory=TOKENIZER_DIR,
        tokenizer_name=TOKENIZER_NAME,
    )

    vocab_size = tokenizer.get_vocab_size()

    model_config = GPTConfig(
        T=args.context_length,
        C=args.n_embed,
        vocab_size=vocab_size,
        num_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )

    model = GPT(config=model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        optimizer_steps=args.optimizer_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        vocab_size=vocab_size,
        context_length=args.context_length,
        stride=args.stride if args.stride is not None else args.context_length,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        run_dir=run_dir,
        args=args
    )