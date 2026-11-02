#!/usr/bin/env bash
# Stage 5: the random-expert control of Stage 4. Did MobileMoE-S's pretrained experts contribute?
#
#   bash scripts/stage5_run.sh        # 15M tokens; about 14 h on the M5 (Stage 4: 13.3 h of training)
#
# Stage 4 exactly (configs/stages/stage4_spread_15m.yaml): Gemma 3 1B, layout spread (20 MoE layers, Gemma's
# FFN kept in 6), the same 35.4M trainable W_in / W_out, the same 15M tokens in the same order, the same
# recipe, micro-batch and evaluation. One change: the 20 MoE blocks keep MobileMoE-S's architecture, but
# every weight is random (each matrix drawn at the scale of the pretrained one it replaces) and frozen.
#
#   0. Checks: the layer map; Stage 4 finished (runs/stage4_spread_15m) and its ridge init is there
#      (runs/stage4_spread_init/init.pt: Stage 5 starts from its W_in); access to MobileMoE-S (its
#      architecture and the scale of its tensors are read from it).
#   1. Micro-batch: Stage 4's (runs/stage4_spread_15m/speed.json, 8 x 2), so both runs split every
#      optimizer step the same way. Delete runs/stage5_random_15m/speed.json to measure again.
#   2. Ridge init (Stage 4's W_in, W_out fitted through the random blocks: a few minutes), then training,
#      with a check on 100k held-out tokens every 100 steps, as Stage 4.
#   3. Final evaluation (250k tokens per set), samples, results/runs.csv.
#   4. Validation: the random blocks verified, the Q/A test and the report against Stage 4
#      (results/stage5_compare.md); scripts/stage5_validate.sh does this step alone.
#
# Pause: Ctrl-C (or `touch runs/stage5_random_15m/PAUSE`). Resume: the same command. Everything printed
# also goes to notes/stage5_console.log. Keep the Mac on its charger with the lid open.
source "$(dirname "$0")/env.sh"
CONFIGS=(configs/base.yaml configs/stages/stage5_random_15m.yaml)
RUN=runs/stage5_random_15m
REF=runs/stage4_spread_15m
CFG_ARGS=()
for c in "${CONFIGS[@]}"; do CFG_ARGS+=(--config "$c"); done
LOG=notes/stage5_console.log
mkdir -p "$RUN" notes

exec > >(tee -i -a "$LOG") 2>&1
echo "== $(date '+%Y-%m-%d %H:%M:%S') stage5_run.sh -> $RUN =="
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dims -w $$ & fi
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"
STOP=0
trap 'STOP=1' INT TERM
paused() { echo "== paused. Run 'bash scripts/stage5_run.sh' to continue where it stopped. =="; exit 0; }

# 0. Checks
echo "== 0/4 checks =="
$PY -m transplant layout "${CFG_ARGS[@]}"
[ -f "$REF/eval.json" ] || { echo "Stage 5 is compared with Stage 4: run 'bash scripts/stage4_run.sh 15m' first ($REF/eval.json is missing)"; exit 1; }
[ -f runs/stage4_spread_init/init.pt ] || { echo "runs/stage4_spread_init/init.pt is missing: Stage 5 starts from Stage 4's ridge W_in"; exit 1; }
set +e
$PY - <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download("facebook/MobileMoE-S-Base", "config.json")
except Exception as e:
    print(f"[stage5] cannot read facebook/MobileMoE-S-Base on Hugging Face ({type(e).__name__}). Stage 5 takes "
          "its architecture and the scale of its tensors from it; log in with `uv run hf auth login`.")
    sys.exit(1)
print("[stage5] MobileMoE-S-Base: access OK (architecture and tensor scale only; every weight is redrawn)")
PY
code=$?
set -e
[ $code -eq 0 ] || exit $code

# 1. Micro-batch: Stage 4's, measured once on this Mac with the same shapes
if [ ! -f "$RUN/speed.json" ] && [ -f "$REF/speed.json" ]; then
  cp "$REF/speed.json" "$RUN/speed.json"
  echo "== 1/4 micro-batch: Stage 4's ($REF/speed.json) =="
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

# 2 and 3. Ridge init, training and the final evaluation; resumes after a pause or an error.
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
echo "== 4/4 validation: random blocks, Q/A test and the Stage 5 report =="
set +e
bash scripts/stage5_validate.sh "$RUN"
code=$?
set -e
[ "$STOP" = 1 ] && paused
[ $code -eq 0 ] || { echo "validation stopped (exit code $code); run 'bash scripts/stage5_validate.sh $RUN'"; exit $code; }
echo "== Stage 5 done: $RUN/eval.json, $RUN/samples.md, $RUN/qa.md, results/stage5_compare.md =="
