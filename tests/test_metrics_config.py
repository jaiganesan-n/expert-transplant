import math

import torch

from transplant.config import load_config
from transplant.eval.cka import linear_cka
from transplant.eval.metrics import gap_recovery, summarize_ppl
from transplant.eval.routing import chance_jaccard, routing_health
from transplant.models.layer_map import map_layers, middle_out, parse_layers
from transplant.train.optim import lr_lambda, steps_for_tokens


def test_gap_recovery():
    assert gap_recovery(30.0, 20.0, 20.0) == 1.0
    assert gap_recovery(30.0, 30.0, 20.0) == 0.0
    assert gap_recovery(30.0, 25.0, 20.0) == 0.5
    s = summarize_ppl(math.log(20), math.log(30), math.log(25))
    assert abs(s["gr"] - 0.5) < 1e-9 and abs(s["ppl_ratio"] - 1.25) < 1e-9


def test_chance_jaccard_for_mobilemoe_routing():
    assert abs(chance_jaccard(60, 4) - 0.039) < 0.001


def test_routing_health():
    uniform = torch.arange(600).remainder(60).reshape(150, 4)
    h = routing_health(uniform, 60)
    assert abs(h["normalized_entropy"] - 1.0) < 1e-9 and h["dead_experts"] == 0
    collapsed = torch.tensor([[0, 1, 2, 3]] * 100)
    h2 = routing_health(collapsed, 60)
    assert h2["normalized_entropy"] < 0.4 and h2["dead_experts"] == 56


def test_cka_identity_and_rotation():
    x = torch.randn(300, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert abs(linear_cka(x, x) - 1) < 1e-9
    assert abs(linear_cka(x, x @ q) - 1) < 1e-9


def test_layer_specs_and_maps():
    assert middle_out(26, 4) == [11, 12, 13, 14]
    assert middle_out(26, 8) == list(range(9, 17))
    assert parse_layers("all", 26) == list(range(26))
    assert parse_layers("first:8", 26) == list(range(8))
    assert parse_layers("last:8", 26) == list(range(18, 26))
    assert map_layers([0, 13, 25], 26, 26) == {0: 0, 13: 13, 25: 25}
    assert map_layers([20], 40, 32) == {20: 16}  # Granite pair, plan's E1 example
    assert map_layers([13], 26, 26, "offset:2") == {13: 15}
    assert map_layers([3, 20], 26, 26, "fixed:13") == {3: 13, 20: 13}


def test_schedules_and_budgets():
    # Rounded down: a run never reads more than its budget.
    assert steps_for_tokens(15_000_000, 1, 16, 1024) == 915   # Stage 2 end to end, 16,384 tokens per step
    assert steps_for_tokens(750_000, 1, 16, 1024) == 45       # Stage 1 end to end
    assert steps_for_tokens(250_000, 2, 4, 1024) == 30        # Stage 1 local matching
    assert steps_for_tokens(10, 1, 16, 1024) == 1
    f = lr_lambda("cosine", 100, 2, 0.1)
    assert f(0) == 0.5 and abs(f(100) - 0.1) < 1e-9
    g = lr_lambda("constant", 1000, 100)
    assert g(99) == 1.0 and g(500) == 1.0


def test_config_overrides(tmp_path):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config([repo / "configs" / "base.yaml", repo / "configs" / "stages" / "stage2_full.yaml"],
                      ["e2e.lr=1.0e-4", "run.tokens=40000000"])
    assert cfg.e2e.lr == 1e-4 and cfg.run.tokens == 40_000_000
    assert cfg.pair.layers == "all" and cfg.device == "mps"            # M5 defaults
    assert cfg.precision.a_dtype == "bfloat16" and not cfg.precision.grad_scaler
    assert cfg.optim.name == "adamw" and list(cfg.optim.betas) == [0.9, 0.95]


def test_stage_budgets():
    from pathlib import Path

    from transplant.train.run import token_budget

    repo = Path(__file__).resolve().parents[1]
    s1 = load_config([repo / "configs" / "base.yaml", repo / "configs" / "stages" / "stage1_test.yaml"])
    s2 = load_config([repo / "configs" / "base.yaml", repo / "configs" / "stages" / "stage2_full.yaml"])
    assert token_budget(s1)[:2] == (250_000, 750_000)          # 1M tokens: 25% local, 75% end to end
    assert token_budget(s2)[:2] == (5_000_000, 15_000_000)     # 20M tokens
    ext = load_config([repo / "configs" / "base.yaml", repo / "configs" / "stages" / "stage2_full.yaml"],
                      ["run.extend_from=runs/stage2_full", "run.tokens=20000000"])
    local, e2e, frm = token_budget(ext)
    assert (local, e2e, str(frm)) == (0, 20_000_000, "runs/stage2_full")
