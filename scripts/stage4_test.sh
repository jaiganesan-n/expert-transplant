#!/usr/bin/env bash
# Stage 4 tests, before any training (about 15 to 25 min; the first run downloads MobileMoE-S, ~2.6 GB):
#
#   bash scripts/stage4_test.sh
#
#   1. The layer map and parameter count of the spread layout (no model loaded).
#   2. Unit tests on tiny stand-in models: layout, configs, a partial transplant that trains,
#      pause and resume.
#   3. MobileMoE-S inspected: module names, expert count, top-k, routed scaling factor
#      (runs/stage4_inspect/m0_inspect.json), checked against configs/pairs/stage4_mobilemoe_s.yaml.
#   4. The gating tests on the real models with Stage 4's configs: no-op surgery, MobileMoE-S block
#      extraction and the bucketed forward against its own, switch-off equals Gemma, gradients
#      only in the projections and the 6 kept FFNs untouched, memory of real training steps,
#      chunked loss, tokenizer alignment.
# Needs: Gemma's and MobileMoE-S-Base's licenses accepted on Hugging Face, `uv run hf auth login`.
source "$(dirname "$0")/env.sh"
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"
CONFIGS="configs/base.yaml,configs/stages/stage4a_spread_1m.yaml"
CFG_ARGS=(--config configs/base.yaml --config configs/stages/stage4a_spread_1m.yaml)

echo "== 1/4 layout =="
$PY -m transplant layout "${CFG_ARGS[@]}"

echo "== 2/4 unit tests (tiny models) =="
$PY -m pytest -q tests/test_stage4.py tests/test_interrupt.py tests/test_stage3.py

echo "== 3/4 MobileMoE-S: what the code finds inside it =="
$PY -m transplant inspect "${CFG_ARGS[@]}" --out runs/stage4_inspect > runs/stage4_inspect.log 2>&1 \
  || { tail -20 runs/stage4_inspect.log; exit 1; }
$PY - <<'PY'
import json, sys
import yaml
rep = json.load(open("runs/stage4_inspect/m0_inspect.json"))
pair = yaml.safe_load(open("configs/pairs/stage4_mobilemoe_s.yaml"))["pair"]["b"]
b, cfg = rep["b"], rep["b"].get("config", {})
checks = {
    "layers": (b.get("n_layers"), pair["num_layers"]),
    "hidden size": (cfg.get("hidden_size"), pair["hidden_size"]),
    "MoE module": (b.get("moe_attr"), pair["moe_attr"]),
    "norm in front of it": (b.get("norm_attr"), pair["norm_attr"]),
    "router": (b.get("router_attr"), pair["router_attr"]),
    "shared expert": (b.get("shared_attr"), pair["shared_attr"]),
}
for key in ("num_experts", "n_routed_experts", "num_local_experts"):
    if key in cfg:
        checks["routed experts"] = (cfg[key], pair["num_experts"])
for key in ("num_experts_per_tok", "top_k", "moe_top_k"):
    if key in cfg:
        checks["experts per token"] = (cfg[key], pair["top_k"])
bad = []
for name, (found, want) in checks.items():
    ok = found == want
    bad += [] if ok else [name]
    print(f"  {'ok ' if ok else 'BAD'} {name}: found {found}, config {want}")
extra = {k: cfg[k] for k in cfg if any(s in k for s in ("scaling", "intermediate", "shared", "act", "router"))}
print("  MobileMoE-S settings to note [S4-A1]:", json.dumps(extra))
if bad:
    print(f"[stage4] fix configs/pairs/stage4_mobilemoe_s.yaml: {', '.join(bad)}")
    sys.exit(1)
PY

echo "== 4/4 gating tests on the real models (Gemma 3 1B + MobileMoE-S, layout spread) =="
TRANSPLANT_REAL=1 TRANSPLANT_REAL_CONFIG="$CONFIGS" $PY -m pytest -v tests/test_real_models.py
echo "== Stage 4 tests passed: start training with 'bash scripts/stage4_run.sh 4a' =="
