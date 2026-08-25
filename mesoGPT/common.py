from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_DIR = ROOT_DIR / "artifacts" / "tokenizer"
TOKENIZER_NAME = "tok-v16384-c450.00m"  # Do not include .pkl

CHECKPOINT_DIR = ROOT_DIR / "weights"
MODEL_NAME = "model" # Do not include .pt