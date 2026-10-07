#!/usr/bin/env bash
# One-time setup on the Mac: Python 3.12 environment with uv, tests, the tiny end-to-end run.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv)"
  exit 1
fi
uv sync                                   # creates .venv from uv.lock (torch 2.8.0, transformers 4.57.6, pytest)
uv run python -c "import torch; print('torch', torch.__version__, '| MPS available:', torch.backends.mps.is_available())"
uv run pytest -q                          # tiny stand-in models, CPU, about 20 seconds
uv run python -m transplant smoke         # whole pipeline on tiny models, about a minute
echo "Next: uv run hf auth login   (accept the Gemma 3 and MobileMoE-M terms on Hugging Face first)"
