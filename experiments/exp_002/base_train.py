import torch
import torch.nn as nn

from dataclasses import asdict
from pathlib import Path
import argparse
import math
import json
import time

from model import GPT, GPTConfig

from mesoGPT.dataloader import create_dataloader
from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import TOKENIZER_DIR, TOKENIZER_NAME


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# AMP is enabled only for CUDA devices.
amp_enabled = device.type == "cuda"

# This prefers BF16 when supported and otherwise uses FP16.
amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16)

@torch.no_grad()
def estimate_loss(model, tokenizer,criterion, 
                  eval_iters, batch_size, vocab_size, context_length, 
                  stride, num_workers):

    model.eval()
    losses = {}

    train_dataloader_eval = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        repeat=False,
        drop_last=False,
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

        total_loss = 0.0
        total_bytes = 0
        total_tokens = 0

        for batch_index, (xb, yb, num_bytes, yb_length) in enumerate(dataloader):

            if eval_iters is not None and batch_index >= eval_iters:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):

                logits = model(xb)

                loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

            batch_tokens = yb_length.sum().item()
            batch_bytes = num_bytes.sum().item()
            
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_bytes += batch_bytes

        if total_tokens == 0:
            raise RuntimeError(
                f"No evaluation batches were produced for split {split!r}."
            )
            
        losses[split] = {
            "loss": total_loss / total_tokens,
            "bpb": total_loss / (math.log(2) * total_bytes),
        }

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
          micro_batch_size, gradient_accumulation_steps, 
          vocab_size, context_length, stride, 
          num_workers, run_dir, args, scalar):

    assert eval_interval > 0, "eval_interval must be greater than 0 to avoid division by zero error."
    
    model.train()
    train_dataloader = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=micro_batch_size,
        repeat=True,
        drop_last=True,
        num_workers=num_workers,
    )

    train_iterator = iter(train_dataloader)

    best_val_loss = float("inf")

    for step in range(optimizer_steps):

        lr = getlr(step, args.max_lr_schedule_steps, args.warmup_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- start of optimization step ---
        # if device.type == "cuda":
        #     torch.cuda.synchronize()
        # t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        for batch_index in range(gradient_accumulation_steps):

            xb, yb, num_bytes, yb_length = next(train_iterator)

            # Asynchronous
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):

                logits = model(xb)      

                loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

                loss = loss / gradient_accumulation_steps

            scalar.scale(loss).backward()

        scalar.step(optimizer)
        scalar.update()

        # if device.type == "cuda":
        #     torch.cuda.synchronize()
        # dt = time.perf_counter() - t0
        # --- end of optimization step ---

        #print(f"Step: {step+1}, Time: {dt:.2f}s")

        completed_steps = step + 1

        if completed_steps % eval_interval == 0 or completed_steps == optimizer_steps:

            # --- start of evaluation step ---
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            # t0 = time.perf_counter()

            losses = estimate_loss(
                model=model,
                tokenizer=tokenizer,
                criterion=criterion,
                eval_iters=eval_iters,
                batch_size=micro_batch_size,
                vocab_size=vocab_size,
                context_length=context_length,
                stride=stride,
                num_workers=num_workers
            )

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            # dt = time.perf_counter() - t0
            # --- end of evaluation step ---

            #print(f"Evaluation Time: {dt:.2f}s")

            print(f"Step: {completed_steps}: "  
                f"Train Loss: {losses['train']['loss']:.4f}, "
                f"Train bpb: {losses['train']['bpb']:.4f}, "
                f"Val Loss: {losses['val']['loss']:.4f}, "
                f"Val bpb: {losses['val']['bpb']:.4f}")
            
            if losses['val']['loss'] < best_val_loss:

                best_val_loss = losses['val']['loss']

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
        print(f"Step: {completed_steps} completed")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GPT model.")

    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model.")
    parser.add_argument("--n_embed", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate.")

    parser.add_argument("--global_batch_size", type=int, default=64, help="Global batch size per optimizer step.")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Number of sequences processed at once on each GPU. Defaults to global_batch_size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, required=False, help="Learning rate for the optimizer.")
    parser.add_argument("--optimizer_steps", type=int, default=5000, help="Maximum number of training steps.")

    parser.add_argument("--max_lr", type=float, default=3e-4, required=False, help="Maximum learning rate for the optimizer.")
    parser.add_argument("--min_lr", type=float, default=3e-5, required=False, help="Minimum learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=10, required=False, help="Number of warmup steps for learning rate scheduling.")
    parser.add_argument("--max_lr_schedule_steps", type=int, default=50, required=False, help="Total number of steps for learning rate scheduling.")

    parser.add_argument("--eval_interval", type=int, default=250, help="Interval for evaluation during training.")
    parser.add_argument("--eval_iters", type=int, default=None, required=False, help="Number of iterations for evaluation.")

    parser.add_argument("--stride", type=int, default=None, help="Stride for the dataset. Defaults to context_length if not provided.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for the dataloader.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory where all artifacts for this run are saved.",
)
    args = parser.parse_args()

    if args.micro_batch_size is None:
        args.micro_batch_size = args.global_batch_size

    if args.global_batch_size <= 0:
        parser.error("--global_batch_size must be greater than zero.")

    if args.micro_batch_size <= 0:
        parser.error("--micro_batch_size must be greater than zero.")

    if args.global_batch_size % args.micro_batch_size != 0:
        parser.error(
            "--global_batch_size must be divisible by --micro_batch_size."
        )

    args.gradient_accumulation_steps = (args.global_batch_size // args.micro_batch_size)

    print(f"Global batch size: {args.global_batch_size}")
    print(f"Micro-batch size: {args.micro_batch_size}")
    print(
        "Gradient accumulation steps: "
        f"{args.gradient_accumulation_steps}"
    )

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

    model = GPT(config=model_config).to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # The scaler is active for FP16 and disabled for BF16.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    assert args.global_batch_size == (
        args.micro_batch_size * args.gradient_accumulation_steps
    )

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        optimizer_steps=args.optimizer_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        vocab_size=vocab_size,
        context_length=args.context_length,
        stride=args.stride if args.stride is not None else args.context_length,
        num_workers=args.num_workers,
        run_dir=run_dir,
        args=args,
        scalar=scaler
    )