import torch
from mesoGPT.model import GPT
from mesoGPT.tokenizer import BPETokenizer
from mesoGPT.common import TOKENIZER_DIR, TOKENIZER_NAME, CHECKPOINT_DIR, MODEL_NAME

text = " "
max_tokens = 1000
temperature = 1.0


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    model.eval()
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)  # (1, T) because the model expects (B, T)
    output = model.generate(idx, max_tokens=max_tokens, temperature=temperature)
    token_ids = output[0].cpu().tolist()  # Move to CPU and convert to list
    return tokenizer.decode(token_ids)   # output is (1, T), so we access the first element and convert it to a list of integers before decoding.

if __name__ == "__main__":
    ckpt = torch.load(CHECKPOINT_DIR / f"{MODEL_NAME}.pt", map_location=device, weights_only=True)
    model = GPT(**ckpt['model_args']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    tokenizer = tokenizer = BPETokenizer.from_directory(
        tokenizer_directory=TOKENIZER_DIR,
        tokenizer_name=TOKENIZER_NAME,  
    )
    print(generate_text(model, tokenizer, prompt=text, max_tokens=max_tokens, temperature=temperature))