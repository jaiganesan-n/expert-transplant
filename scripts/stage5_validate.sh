#!/usr/bin/env bash
# Stage 5 validation of a finished run: the random blocks verified, the Q/A test, the report against Stage 4.
#
#   bash scripts/stage5_validate.sh                            # runs/stage5_random_15m
#   bash scripts/stage5_validate.sh runs/stage5_random_15m
#
# * scripts/verify_stage5_random.py RUN (a few minutes): the blocks the run trained with have MobileMoE-S's
#   architecture and no pretrained value (each matrix at the pretrained scale), are frozen, are the ones
#   training used (fingerprint), and the run started from Stage 4's W_in. Written to RUN/verify_random.txt.
# * Q/A (`transplant ask --run RUN --ablated`): 30 questions with one-sentence answers from Gemma, the hybrid
#   with random experts and the layout floor. RUN/qa.md and RUN/qa.json; skipped when qa.json is up to date.
# * Report: results/stage5_compare.md (verdict, text, learning curves, ridge init, Q/A, routing).
source "$(dirname "$0")/env.sh"
RUN=${1:-runs/stage5_random_15m}
[ -f "$RUN/projections.pt" ] || { echo "$RUN has no projections.pt: train it first (scripts/stage5_run.sh)"; exit 1; }
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"

echo "== random blocks: verify ($RUN) =="
if ! $PY scripts/verify_stage5_random.py "$RUN" | tee "$RUN/verify_random.txt"; then
  echo "verification failed: see $RUN/verify_random.txt"
  exit 1
fi
if grep -q '"scoring": "sentence_v1"' "$RUN/qa.json" 2>/dev/null; then
  echo "== Q/A: $RUN/qa.json is up to date =="
else
  echo "== Q/A: Gemma, hybrid with random experts and layout floor ($RUN) =="
  $PY -m transplant ask --run "$RUN" --ablated
fi
echo "== Stage 5 report =="
$PY scripts/stage5_compare.py
