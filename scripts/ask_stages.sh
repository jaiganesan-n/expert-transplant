#!/usr/bin/env bash
# Question-and-answer test, one-sentence answers, for every stage: Gemma (original), the hybrid trained in
# that stage, and No FFN (Gemma with all 26 FFNs deleted). Then all stages side by side in
# results/qa_compare.md. Nothing is trained; about 3 to 5 minutes per stage.
#
#   bash scripts/ask_stages.sh                    # runs/stage1_test, runs/stage2_5m, and runs/stage3* once it exists
#   bash scripts/ask_stages.sh runs/a runs/b ...  # any runs, earliest first
#   FORCE=1 bash scripts/ask_stages.sh            # ask again even where qa.json is already up to date
#
# A stage whose qa.json comes from this version of the test is not asked again, so after Stage 3
# the same command asks only Stage 3 and compares all three.
source "$(dirname "$0")/env.sh"
VERSION='"scoring": "sentence_v1"'   # SCORING in src/transplant/eval/ask.py

RUNS=()
if [ $# -gt 0 ]; then
  RUNS=("$@")
else
  for d in runs/stage1_test runs/stage2_5m runs/stage3*; do
    if [ -f "$d/projections.pt" ]; then RUNS+=("$d"); fi
  done
fi
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "No trained runs found (runs/<name>/projections.pt)."
  exit 1
fi

for RUN in "${RUNS[@]}"; do
  if [ "${FORCE:-0}" != 1 ] && grep -q "$VERSION" "$RUN/qa.json" 2>/dev/null; then
    echo "== $RUN: qa.json is up to date, not asked again (FORCE=1 to redo) =="
    continue
  fi
  echo "== $RUN: Gemma, hybrid and No FFN =="
  uv run transplant ask --run "$RUN" --ablated
done

echo "== ${RUNS[*]} side by side =="
uv run transplant ask-compare "${RUNS[@]}"
