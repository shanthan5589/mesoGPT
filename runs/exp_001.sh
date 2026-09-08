#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with:
# bash runs/exp_001.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ ! -x /usr/bin/time ]]; then
    echo "/usr/bin/time not found. Run: sudo apt update && sudo apt install -y time" >&2
    exit 1
fi

VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Virtual environment not found. Run: bash setup.sh" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

RUN_TIMESTAMP="$(date -u +'%Y-%m-%d_%H-%M-%S_UTC')"
RUNS_DIR="$PROJECT_ROOT/experiments/exp_001/runs"
RUN_DIR="$RUNS_DIR/$RUN_TIMESTAMP"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/tokenizer.log"

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
    /usr/bin/time -v python experiments/exp_001/tok_train.py \
        --max-training-chars "$TRAINING_CHARS" \
        --max-chars-per-document 10_000 \
        --vocab-size 16384 \
        --tokenizer-output-directory "$RUN_DIR"
done

echo "===== EXPERIMENT COMPLETE ====="
echo "Log saved to: $LOG_FILE"