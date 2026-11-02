#!/usr/bin/env bash
# Stage 5 tests, before any training (about 10 min; MobileMoE-S and Gemma are already in the cache from Stage 4):
#
#   bash scripts/stage5_test.sh
#
#   1. The layer map and parameter count of the layout (no model loaded): Stage 4's.
#   2. Unit tests on tiny stand-in models: the Stage 5 config is Stage 4's but for the experts; the random
#      draw (no pretrained value left, the pretrained scale, the same on the CPU and the GPU); the start from
#      Stage 4's W_in; the report. Plus the Stage 4 and block-extraction tests it builds on.
#   3. The random blocks on the real models (scripts/verify_stage5_random.py with the Stage 5 config).
#   4. Memory and speed of real Stage 5 training steps at Stage 4's micro-batch, in a fresh process.
# Needs: Gemma's and MobileMoE-S-Base's licenses accepted on Hugging Face, `uv run hf auth login`.
source "$(dirname "$0")/env.sh"
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"
CFG_ARGS=(--config configs/base.yaml --config configs/stages/stage5_random_15m.yaml)

echo "== 1/4 layout =="
$PY -m transplant layout "${CFG_ARGS[@]}"

echo "== 2/4 unit tests (tiny models) =="
$PY -m pytest -q tests/test_stage5.py tests/test_stage4.py tests/test_extraction.py

echo "== 3/4 the random blocks on the real models =="
$PY scripts/verify_stage5_random.py "${CFG_ARGS[@]}"

echo "== 4/4 memory and speed of training steps at Stage 4's micro-batch =="
read -r MB ACCUM < <($PY -c "import json; d = json.load(open('runs/stage4_spread_15m/speed.json')); print(d['micro_batch'], d['grad_accum'])" 2>/dev/null || echo "8 2")
$PY -m transplant memcheck --e2e-only "${CFG_ARGS[@]}" --set e2e.micro_batch="$MB" --set e2e.grad_accum="$ACCUM"
echo "== Stage 5 tests passed: start training with 'bash scripts/stage5_run.sh' =="
