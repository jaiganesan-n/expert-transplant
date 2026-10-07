#!/usr/bin/env bash
# Stage 2: full run. All 26 layers, 20M training tokens, each read once.
# Re-running resumes from the last checkpoint; a finished run is skipped.
source "$(dirname "$0")/env.sh"
$KEEP_AWAKE uv run python -m transplant train --resume $BASE --config configs/stages/stage2_full.yaml "$@"
uv run python -m transplant summarize
echo "If loss is still falling and accuracy still rising, add tokens: bash scripts/stage2_extend.sh 20000000"
