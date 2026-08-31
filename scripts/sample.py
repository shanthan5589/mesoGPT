import argparse
import json
from pathlib import Path

import torch

from mesoGPT.common import ROOT_DIR, TOKENIZER_DIR, TOKENIZER_NAME
from mesoGPT.model_registry import build_model
from mesoGPT.tokenizer import BPETokenizer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


def resolve_run_directory(run_dir):

    run_dir = run_dir.expanduser()

    if not run_dir.is_absolute():
        run_dir = ROOT_DIR / run_dir

    return run_dir.resolve()


def load_run(run_dir):
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    model_path = run_dir / "model.pt"
    metadata_path = run_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    exp_no = metadata.get("exp_no")
    model_config = metadata.get("model_config")

    if not exp_no:
        raise ValueError(
            f"exp_no is missing from {metadata_path}"
        )

    if not isinstance(model_config, dict):
        raise ValueError(
            f"model_config is missing or invalid in {metadata_path}"
        )

    tokenizer_name = metadata.get(
        "tokenizer_name",
        TOKENIZER_NAME,
    )

    tokenizer = BPETokenizer.from_directory(
        tokenizer_directory=TOKENIZER_DIR,
        tokenizer_name=tokenizer_name,
    )

    expected_vocab_size = model_config.get("vocab_size")

    if expected_vocab_size is None:
        raise ValueError(
            "vocab_size is missing from model_config"
        )

    if tokenizer.get_vocab_size() != expected_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match the model: "
            f"{tokenizer.get_vocab_size()} != "
            f"{expected_vocab_size}"
        )

    model = build_model(
        exp_no=exp_no,
        model_config=model_config,
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, tokenizer, metadata

@torch.no_grad()
def generate_text(model, tokenizer, prompt="Hello, my dog is cute", max_tokens=100, temperature=1.0):
    model.eval()
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)  # (1, T) because the model expects (B, T)
    output = model.generate(idx, max_tokens=max_tokens, temperature=temperature)
    token_ids = output[0].cpu().tolist()  # Move to CPU and convert to list
    return tokenizer.decode(token_ids)   # output is (1, T), so we access the first element and convert it to a list of integers before decoding.


def main():

    parser = argparse.ArgumentParser(
        description="Generate text from any mesoGPT experiment."
    )

    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing model.pt and model_meta.json.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. If omitted, prompt interactively.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )

    args = parser.parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    run_dir = resolve_run_directory(args.run_dir)

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")

    model, tokenizer, metadata = load_run(run_dir)

    print(f"Experiment no: {metadata['exp_no']}")
    print(f"Checkpoint step: {metadata.get('step', 'unknown')}")

    prompt = args.prompt

    if prompt is None:
        prompt = input("Enter your prompt: ")

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(generated_text)


if __name__ == "__main__":
    main()