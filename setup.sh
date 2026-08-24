#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

eval "$(conda shell.bash hook)"
conda env create --name mesogpt --file environment.yml
conda activate mesogpt
python -m pip install -e .

echo "Setup complete."
