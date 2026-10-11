#!/usr/bin/env bash
# Stage 4 validation of a finished run: the Q/A test and the report against Stages 1-3.
#
#   bash scripts/stage4_validate.sh runs/stage4a_spread_1m               # Q/A (if not done) + report
#   bash scripts/stage4_validate.sh runs/stage4a_spread_1m --agreement   # also routing agreement (~40 min)
#
# * Q/A (`transplant ask --run RUN --ablated`): 30 questions with one-sentence answers from Gemma,
#   the hybrid and the layout floor (Gemma with the 20 transplanted FFNs deleted, the 6 kept).
#   Writes RUN/qa.md and RUN/qa.json; skipped when qa.json is already up to date.
# * --agreement: re-evaluates the run and measures how often the hybrid picks the experts
#   MobileMoE-S itself would pick (mean Jaccard of the top-4 sets; chance 0.039). RUN/eval_full.json.
# * Report: results/stage4_compare.md (text, matched tokens, Q/A with both floors, routing).
source "$(dirname "$0")/env.sh"
RUN=${1:?usage: bash scripts/stage4_validate.sh runs/<stage4 run> [--agreement]}
[ -f "$RUN/projections.pt" ] || { echo "$RUN has no projections.pt: train it first (scripts/stage4_run.sh)"; exit 1; }
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"

if grep -q '"scoring": "sentence_v1"' "$RUN/qa.json" 2>/dev/null; then
  echo "== Q/A: $RUN/qa.json is up to date =="
else
  echo "== Q/A: Gemma, hybrid and layout floor ($RUN) =="
  $PY -m transplant ask --run "$RUN" --ablated
fi
if [ "${2:-}" = "--agreement" ]; then
  echo "== routing agreement with MobileMoE-S's own routing (re-evaluates the run) =="
  $PY -m transplant eval --config "$RUN/config.yaml" --routing-agreement
fi
echo "== Stage 4 report =="
$PY scripts/stage4_compare.py
