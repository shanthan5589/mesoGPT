#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with:
# bash runs/tokenizer_experiment.sh

eval "$(conda shell.bash hook)"
conda activate gpu_env

LOG_FILE="tokenizer_experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

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
    /usr/bin/time -v python -m scripts.tok_train \
        --max-training-chars "$TRAINING_CHARS" \
        --max-characters-per-document 10_000 \
        --vocab-size 16384
done

echo "===== EXPERIMENT COMPLETE ====="
echo "Log saved to: $LOG_FILE"