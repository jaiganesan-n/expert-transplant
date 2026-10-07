#!/usr/bin/env bash
# Stage 1: test run. All 26 layers, 1M training tokens, each read once.
# Re-running resumes from the last checkpoint; a finished run is skipped.
source "$(dirname "$0")/env.sh"
$KEEP_AWAKE uv run python -m transplant train --resume $BASE --config configs/stages/stage1_test.yaml "$@"
uv run python -m transplant summarize
# Measured speed and memory, and what they mean for Stage 2 (5M local-matching + 15M end-to-end tokens).
uv run python - <<'PY'
import json, pathlib
s = json.loads(pathlib.Path("runs/stage1_test/summary.json").read_text())
loc, e2e = s.get("local_tokens_per_s"), s.get("e2e_tokens_per_s")
if loc and e2e:
    print(f"Speed on this Mac: local matching {loc:,.0f} tokens/s, end to end {e2e:,.0f} tokens/s; "
          f"peak memory {s.get('peak_gb') or 0:.1f} GB")
    print(f"Stage 2 at this speed: about {(5e6 / loc + 15e6 / e2e) / 3600:.1f} hours of training, "
          "plus ridge init and evaluation")
PY
echo "Read runs/stage1_test/samples.md, runs/stage1_test/eval.json and runs/stage1_test/metrics.jsonl"
