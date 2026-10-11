#!/usr/bin/env bash
# Stage 3: all 26 layers, 20M training tokens in total, end to end only, then the Q/A test.
#
#   bash scripts/stage3_run.sh          # start, or continue after any pause or stop
#
# It continues Stage 2 (runs/stage2_5m: its projections and its data stream, 5M tokens) for 15M
# more tokens, so no token is read twice and ridge init and local matching are not run again.
# Same objective and settings as Stage 2; configs/stages/stage3_20m.yaml lists the few that change.
#
#   1. Speed check, once (about 10 to 15 min): micro-batch 1, 2, 4 and 8 measured on this Mac;
#      the fastest that fits in memory is kept in runs/stage3_20m/speed.json. Every optimizer
#      step still sees 16 sequences (16,384 tokens), so only the speed changes.
#   2. Training: 915 steps, a check on 100k held-out tokens every 100 steps, a checkpoint every
#      10 minutes.
#   3. Final evaluation (250k tokens per set), samples, results/runs.csv.
#   4. Q/A test for every stage (Gemma, hybrid, No FFN) and the comparison, results/qa_compare.md.
#
# Pausing and interruptions:
#   * Pause: Ctrl-C (or, from another terminal, `touch runs/stage3_20m/PAUSE`). Training stops at
#     once, saves a checkpoint at the last finished step (seconds) and exits. Resume: run this
#     script again; it continues from that step with the same data. A check or the final
#     evaluation that was cut short continues too (finished evaluation sets are kept).
#   * Sleep (lid closed): the run is frozen and carries on when the Mac wakes.
#   * Shutdown, crash or power cut: at most the last 10 minutes are lost; run this script again.
#     After an error (for example a network drop while streaming data) the script resumes by
#     itself, up to 5 times, a minute apart.
#   * Everything printed also goes to notes/stage3_console.log.
source "$(dirname "$0")/env.sh"
FROM=${FROM:-runs/stage2_5m}
CONFIG=${CONFIG:-configs/stages/stage3_20m.yaml}
RUN=${RUN:-runs/stage3_20m}
LOG=notes/stage3_console.log
[ -f "$FROM/projections.pt" ] || { echo "Stage 2 is not finished: $FROM/projections.pt is missing"; exit 1; }
mkdir -p "$RUN" notes

# Everything also goes to the log; `tee -i` keeps logging through Ctrl-C.
exec > >(tee -i -a "$LOG") 2>&1
echo "== $(date '+%Y-%m-%d %H:%M:%S') stage3_run.sh =="
# Keep the Mac awake while this script runs (a closed lid still sleeps it: keep it open, on the charger).
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dims -w $$ & fi

# The project's Python directly (the editable install sees every code change): Ctrl-C then
# reaches training without a launcher in between.
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"

STOP=0
trap 'STOP=1' INT TERM
paused() { echo "== paused. Run 'bash scripts/stage3_run.sh' to continue where it stopped. =="; exit 0; }

# 1. Speed check (once; measurements finished before a stop are kept).
if [ ! -f "$RUN/speed.json" ]; then
  echo "== 1/4 speed check =="
  set +e
  $PY scripts/stage3_speed.py "$RUN" --config "$FROM/config.yaml" --config "$CONFIG"
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
  $PY -m transplant train --resume --config "$FROM/config.yaml" --config "$CONFIG" \
    --set e2e.micro_batch="$MB" --set e2e.grad_accum="$ACCUM"
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

# 4. Q/A test for every stage (stages already tested are skipped).
echo "== 4/4 Q/A test, every stage =="
set +e
bash scripts/ask_stages.sh
code=$?
set -e
[ "$STOP" = 1 ] && paused
[ $code -eq 0 ] || { echo "Q/A test stopped (exit code $code); run this script again to finish it"; exit $code; }
echo "== Stage 3 done: $RUN/eval.json, $RUN/samples.md, results/qa_compare.md =="
