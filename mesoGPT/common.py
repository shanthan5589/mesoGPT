from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = ROOT_DIR / "experiments"

TOKENIZER_DIR = ROOT_DIR / "artifacts" / "tokenizer"
TOKENIZER_NAME = "tok-v16384-c450.00m"  # Do not include .pkl