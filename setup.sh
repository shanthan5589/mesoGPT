#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

is_supported_python() {
	"$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
	if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
		echo "Python executable not found: $PYTHON_BIN" >&2
		exit 1
	fi
elif is_supported_python python3.12; then
	PYTHON_BIN=python3.12
elif is_supported_python python3.11; then
	PYTHON_BIN=python3.11
elif is_supported_python python3; then
	PYTHON_BIN=python3
else
	echo "mesoGPT requires Python 3.11 or 3.12." >&2
	echo "Install one of those versions, or set PYTHON_BIN to its executable." >&2
	exit 1
fi

if ! is_supported_python "$PYTHON_BIN"; then
	echo "mesoGPT requires Python 3.11 or 3.12; found: $("$PYTHON_BIN" --version)" >&2
	exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
	"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_PYTHON" -m pip install --upgrade pip

if "$VENV_PYTHON" -c 'import torch; raise SystemExit(0 if (2, 4) <= tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 7) else 1)' 2>/dev/null; then
	echo "Using compatible PyTorch already available to this Python installation."
else
	echo "Installing a compatible PyTorch build."
	"$VENV_PYTHON" -m pip install --ignore-installed 'torch>=2.4,<2.7'
fi

"$VENV_PYTHON" -m pip install \
	'pyarrow>=25,<26' \
	'requests>=2.34,<3' \
	'rustbpe==0.1.0' \
	'tiktoken>=0.14,<0.15'
"$VENV_PYTHON" -m pip install --no-deps -e .

"$VENV_PYTHON" - <<'PY'
import torch
import mesoGPT

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
	print("CUDA device: none (CPU mode is available)")
print("mesoGPT import: OK")
PY

echo
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo "Then download data with:"
echo "  python -m mesoGPT.dataset -n 15"