#!/usr/bin/env bash
# Stage 4: Gemma 3 1B keeps 6 FFNs (layers 2, 6, 10, 15, 19, 23); the other 20 are replaced by
# MobileMoE-S MoE blocks at the same relative depth (layout spread). Only W_in / W_out train.
#
#   bash scripts/stage4_run.sh 4a     # 1M tokens: the first run (about 2 to 3 h with ridge init)
#   bash scripts/stage4_run.sh 4b     # 5M tokens, matched to Stage 2 (same tokens, same recipe)
#   bash scripts/stage4_run.sh 4c     # to 20M in total, matched to Stage 3: continues 4b for 15M more
#   bash scripts/stage4_run.sh 15m    # one run of 15M tokens from ridge init (the first 15M tokens of Stages 2-3)
#
# Data: the MobileMoE pre-training mix of Stages 1-3, read in the same order as Stages 2-3, each
# token once. Recipe as Stages 2-3: ridge init (recomputed for MobileMoE-S, shared by 4a and 4b),
# then end to end, CE + 1.0 KL to unmodified Gemma, AdamW, peak LR 3e-4, cosine, 16,384 tokens per step.
#
#   0. Checks: the layer map, and access to MobileMoE-S on Hugging Face.
#   1. Speed check, once per run: micro-batch 1, 2, 4, 8 measured on this Mac (speed.json).
#   2. Training, with a check on 100k held-out tokens every 20 (4a), 50 (4b) or 100 (4c) steps.
#   3. Final evaluation (250k tokens per set), samples, results/runs.csv.
#   4. Validation: Q/A test (Gemma, hybrid, layout floor) and the Stage 4 report against Stages 1-3
#      (results/stage4_compare.md); scripts/stage4_validate.sh does this step alone.
#
# Pause: Ctrl-C (or `touch runs/<run>/PAUSE`). Resume: the same command. Everything printed also
# goes to notes/stage4_console.log. Keep the Mac on its charger with the lid open.
source "$(dirname "$0")/env.sh"
STAGE=${1:-4a}
case "$STAGE" in
  4a) CONFIGS=(configs/base.yaml configs/stages/stage4a_spread_1m.yaml); RUN=runs/stage4a_spread_1m ;;
  4b) CONFIGS=(configs/base.yaml configs/stages/stage4b_spread_5m.yaml); RUN=runs/stage4b_spread_5m ;;
  15m) CONFIGS=(configs/base.yaml configs/stages/stage4_spread_15m.yaml); RUN=runs/stage4_spread_15m ;;
  4c) FROM=runs/stage4b_spread_5m
      if [ ! -f "$FROM/eval.json" ]; then echo "4c continues 4b: run 'bash scripts/stage4_run.sh 4b' first"; exit 1; fi
      CONFIGS=("$FROM/config.yaml" configs/stages/stage4c_spread_20m.yaml); RUN=runs/stage4c_spread_20m ;;
  *) echo "usage: bash scripts/stage4_run.sh 4a|4b|4c|15m"; exit 2 ;;
esac
CFG_ARGS=()
for c in "${CONFIGS[@]}"; do CFG_ARGS+=(--config "$c"); done
LOG=notes/stage4_console.log
mkdir -p "$RUN" notes

exec > >(tee -i -a "$LOG") 2>&1
echo "== $(date '+%Y-%m-%d %H:%M:%S') stage4_run.sh $STAGE -> $RUN =="
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dims -w $$ & fi
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"
STOP=0
trap 'STOP=1' INT TERM
paused() { echo "== paused. Run 'bash scripts/stage4_run.sh $STAGE' to continue where it stopped. =="; exit 0; }

# 0. Checks
echo "== 0/4 checks =="
$PY -m transplant layout "${CFG_ARGS[@]}"
set +e
$PY - <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download("facebook/MobileMoE-S-Base", "config.json")
except Exception as e:
    print(f"[stage4] cannot read facebook/MobileMoE-S-Base on Hugging Face ({type(e).__name__}). Accept its "
          "license on https://huggingface.co/facebook/MobileMoE-S-Base (separate from MobileMoE-M) and log in "
          "with `uv run hf auth login`, then run this again.")
    sys.exit(1)
print("[stage4] MobileMoE-S-Base: access OK")
PY
code=$?
set -e
[ $code -eq 0 ] || exit $code

# 1. Speed check (once per run; 4c reuses 4b's: same models, same memory)
if [ ! -f "$RUN/speed.json" ] && [ "$STAGE" = 4c ] && [ -f "$FROM/speed.json" ]; then
  cp "$FROM/speed.json" "$RUN/speed.json"
  echo "== 1/4 speed check: reusing $FROM/speed.json =="
fi
if [ ! -f "$RUN/speed.json" ]; then
  echo "== 1/4 speed check =="
  set +e
  $PY scripts/stage3_speed.py "$RUN" "${CFG_ARGS[@]}"
  code=$?
  set -e
  [ "$STOP" = 1 ] && paused
  [ $code -eq 0 ] || { echo "speed check failed (exit code $code)"; exit $code; }
fi
read -r MB ACCUM < <($PY -c "import json; d = json.load(open('$RUN/speed.json')); print(d['micro_batch'], d['grad_accum'])")
[ -n "$MB" ] && [ -n "$ACCUM" ] || { echo "no micro-batch in $RUN/speed.json"; exit 1; }

# 2 and 3. Training and the final evaluation; resumes after a pause or an error.
echo "== 2/4 training and 3/4 final evaluation: micro-batch $MB x $ACCUM per step =="
echo "   progress, in another terminal: .venv/bin/python scripts/progress.py $RUN --watch"
echo "   pause: Ctrl-C here, or: touch $RUN/PAUSE"
attempt=1
while true; do
  set +e
  $PY -m transplant train --resume "${CFG_ARGS[@]}" --set e2e.micro_batch="$MB" --set e2e.grad_accum="$ACCUM"
  code=$?
  set -e
  [ $code -eq 0 ] && break
  if [ $code -eq 75 ] || [ $code -eq 130 ] || [ "$STOP" = 1 ]; then paused; fi
  if [ $attempt -ge 5 ]; then
    echo "== training stopped with an error $attempt times (exit code $code); see $LOG. Fix it, then run this script again. =="
    exit $code
  fi
  attempt=$((attempt + 1))
  echo "== error (exit code $code); resuming from the last checkpoint in 60 s, attempt $attempt of 5 (Ctrl-C to stop) =="
  sleep 60 || true
  [ "$STOP" = 1 ] && paused
done
$PY -m transplant summarize

# 4. Validation
echo "== 4/4 validation: Q/A test and the Stage 4 report =="
set +e
bash scripts/stage4_validate.sh "$RUN"
code=$?
set -e
[ "$STOP" = 1 ] && paused
[ $code -eq 0 ] || { echo "validation stopped (exit code $code); run 'bash scripts/stage4_validate.sh $RUN'"; exit $code; }
echo "== Stage $STAGE done: $RUN/eval.json, $RUN/samples.md, $RUN/qa.md, results/stage4_compare.md =="
