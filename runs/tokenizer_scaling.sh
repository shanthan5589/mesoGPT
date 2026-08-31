#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with:
# bash runs/tokenizer_scaling.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/experiments/experiment_001_tokenizer_scaling/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

RUN_TIMESTAMP="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
LOG_FILE="$LOG_DIR/tokenizer_${RUN_TIMESTAMP}.log"

exec > >(tee "$LOG_FILE") 2>&1

eval "$(conda shell.bash hook)"
conda activate mesogpt


echo "===== DATASET DOWNLOAD ====="
/usr/bin/time -v python -m mesoGPT.dataset -n 15

for TRAINING_CHARS in \
    25_000_000 \
    50_000_000 \
    100_000_000 \
    150_000_000 \
    200_000_000 \
    250_000_000 \
    300_000_000 \
    350_000_000 \
    400_000_000 \
    450_000_000 \
    500_000_000
do
    echo "===== TOKENIZER: $TRAINING_CHARS TRAINING CHARACTERS ====="
    /usr/bin/time -v python experiments/tokenizer_scaling/tok_train.py \
        --max-training-chars "$TRAINING_CHARS" \
        --max-chars-per-document 10_000 \
        --vocab-size 16384
done

echo "===== EXPERIMENT COMPLETE ====="
echo "Log saved to: $LOG_FILE"