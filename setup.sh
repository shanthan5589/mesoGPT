#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
TORCH_REQUIREMENT='torch>=2.6,<2.15'
cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
    echo "Create .venv first: python3 -m venv .venv" >&2
    exit 1
fi

case "${1:-}" in
    cpu)
        "$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH_REQUIREMENT"
        ;;
    gpu)
        "$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu126 "$TORCH_REQUIREMENT"
        ;;
    *)
        echo "Usage: bash setup.sh cpu|gpu" >&2
        exit 2
        ;;
esac

"$PYTHON" -m pip install -e .
