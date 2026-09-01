#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

python -m venv --system-site-packages .venv-runpod
source .venv-runpod/bin/activate
python -m pip install -e .
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo "Setup complete. Activate later with: source .venv-runpod/bin/activate"