import torch
import torch.nn as nn

from mesoGPT.model import GPT
from mesoGPT.dataloader import load_data

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
train_ratio = 0.8
stride = context_length
shuffle = True          # Set to False for Validation dataset because what if the randomly picked batch is too easy to predict, if it's too easy we will have a low validation loss and we end up saving a wrong checkpoint.
drop_last = True        # Set to True for Validation dataset
num_workers = 0

# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

vocab, tokenizer, train_dataloader, val_dataloader = load_data(batch_size, context_length, 
                                                               stride, shuffle, 
                                                               drop_last, num_workers, 
                                                               train_ratio=train_ratio)
vocab_size = len(vocab)

model = GPT(vocab_size=vocab_size, T=context_length, 
            C=n_embed, n_layers=n_layers, 
            num_heads=n_heads, dropout=dropout).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def estimate_loss():

    model.eval()
    losses = {}

    for split, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:

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

train_iterator = iter(train_dataloader)
best_val_loss = float('inf')

for step in range(max_steps):
        
    try:
        xb, yb = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_dataloader)
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

        losses = estimate_loss()

        print(f"Step: {completed_steps}: "  
              f"Train Loss: {losses['train']:.4f}, "
              f"Val Loss: {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:

            best_val_loss = losses['val']

            torch.save({
                "model_args": {
                    "T": context_length,
                    "C": n_embed,
                    "vocab_size": vocab_size,
                    "num_heads": n_heads,
                    "n_layers": n_layers,
                    "dropout": dropout
                },
                "vocab": vocab,
                "state_dict": model.state_dict(),
                # Useful if you want to resume training
                "optimizer_state_dict": optimizer.state_dict(),
                "step": completed_steps,
                "val_loss": best_val_loss,
            }, "weights/model.pt")

            print(
                    f"Saved new best checkpoint "
                    f"with validation loss {best_val_loss:.4f}"
                )