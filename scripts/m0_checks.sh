#!/usr/bin/env bash
# M0 on the Mac, after `uv run hf auth login` with both model licenses accepted.
# Downloads Gemma 3 1B (~2 GB) and MobileMoE-M-Base (~5.6 GB) into the Hugging Face cache.
source "$(dirname "$0")/env.sh"
uv run python -m transplant inspect $BASE --out runs/m0       # then pin module names in configs/pairs/primary.yaml
uv run python -m transplant data-check $BASE --docs 3         # every enabled source streams
uv run python -m transplant build-heldout $BASE               # 5M validation + 2M calibration tokens + WikiText-103 + C4
uv run python -m transplant align-stats $BASE --chars 1000000 --out runs/m0
TRANSPLANT_REAL=1 uv run pytest tests/test_real_models.py -v  # gating tests on the M5; nothing trains until they pass
