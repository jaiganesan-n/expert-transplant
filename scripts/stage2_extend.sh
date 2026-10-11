#!/usr/bin/env bash
# Add tokens to a finished Stage 2 run after reading its results: continue its projections and
# data stream for N more tokens (default 20M), end to end, so no token is read twice. The
# extension keeps every setting of the run it extends (its config.yaml).
#   bash scripts/stage2_extend.sh 20000000            # extends the Stage 2 run with the most tokens
#   bash scripts/stage2_extend.sh 5000000 runs/X      # extends runs/X
source "$(dirname "$0")/env.sh"
TOKENS="${1:-20000000}"
# Default: the finished Stage 2 run with the most tokens trained so far.
FROM="${2:-$(uv run python - <<'PY'
import json, pathlib
best = (-1, "")
for d in pathlib.Path("runs").glob("stage2*"):
    s = d / "summary.json"
    if (d / "projections.pt").exists() and s.exists():
        best = max(best, (json.loads(s.read_text()).get("tokens_trained_total", 0), str(d)))
print(best[1])
PY
)}"
[[ -f "$FROM/projections.pt" ]] || { echo "no finished run at '$FROM'"; exit 1; }
DONE=$(uv run python -c "import json; print(json.load(open('$FROM/summary.json'))['tokens_trained_total'])")
NAME="stage2_$(( (DONE + TOKENS) / 1000000 ))M"
echo "Extending $FROM ($DONE tokens) by $TOKENS tokens -> runs/$NAME"
$KEEP_AWAKE uv run python -m transplant train --resume --config "$FROM/config.yaml" \
  --set run.name="$NAME" --set run.extend_from="$FROM" --set run.tokens="$TOKENS"
uv run python -m transplant summarize
