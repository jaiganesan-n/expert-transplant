"""Stage 4: Gemma 3 1B + MobileMoE-S, layout spread (20 MoE layers, 6 Gemma FFNs kept)."""
import json
from pathlib import Path

import pytest
import torch

from transplant.config import load_config
from transplant.data.heldout import build_heldout
from transplant.models import surgery
from transplant.models.layer_map import kept_layers, resolve_layer_map, spread_layers
from transplant.models.tiny import tiny_tokenizer
from transplant.train.optim import steps_for_tokens
from transplant.train.run import _expected_init_shapes, run_training, token_budget

REPO = Path(__file__).resolve().parents[1]
STAGES = REPO / "configs" / "stages"
SPREAD_KEPT = [2, 6, 10, 15, 19, 23]
SPREAD_MAP = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7, 11: 8, 12: 9, 13: 10, 14: 11, 16: 12, 17: 13,
              18: 14, 20: 15, 21: 16, 22: 17, 24: 18, 25: 19}


def test_spread_layout_for_gemma_and_mobilemoe_s():
    lm = spread_layers(26, 20)
    assert lm == SPREAD_MAP  # the map of the Stage 4 context and the infographic
    assert kept_layers(26, list(lm)) == SPREAD_KEPT
    assert sorted(lm.values()) == list(range(20))  # every MobileMoE-S layer once, in order
    for g, j in lm.items():  # the same relative depth both ways
        assert g == round(j * 25 / 19) and j == round(g * 19 / 25)
    a_layers, lm2 = resolve_layer_map({"layers": "spread", "layer_map": "relative_depth"}, 26, 20)
    assert lm2 == lm and a_layers == sorted(lm)
    # the old layouts are unchanged: all 26 layers, A layer i <- B layer i
    assert resolve_layer_map({"layers": "all"}, 26, 26)[1] == {i: i for i in range(26)}
    with pytest.raises(ValueError):
        spread_layers(20, 26)


def _stage4_cfg(name: str):
    return load_config([REPO / "configs" / "base.yaml", STAGES / name])


def test_stage4_configs():
    a, b = _stage4_cfg("stage4a_spread_1m.yaml"), _stage4_cfg("stage4b_spread_5m.yaml")
    for cfg in (a, b):
        pb = cfg["pair"]["b"]
        assert pb["model_id"] == "facebook/MobileMoE-S-Base" and pb["revision"] is None
        assert (pb["num_layers"], pb["hidden_size"], pb["num_experts"], pb["top_k"]) == (20, 768, 60, 4)
        assert cfg["pair"]["layers"] == "spread" and cfg["pair"]["a"]["model_id"] == "google/gemma-3-1b-pt"
        assert cfg["local"]["enabled"] is False and cfg["e2e"]["beta_kl"] == 1.0 and cfg["e2e"]["lr"] == 3.0e-4
        assert cfg["transplant"]["init_path"] == "runs/stage4_spread_init/init.pt"
        assert cfg["eval"]["final_tokens"] == 250_000 and cfg["eval"]["subset_tokens"] == 100_000
        assert cfg["run"]["extend_from"] is None  # data stream from its start, as Stage 2
    assert steps_for_tokens(token_budget(a)[1], 1, 16, 1024) == 61
    assert steps_for_tokens(token_budget(b)[1], 1, 16, 1024) == 305
    c = load_config([REPO / "configs" / "base.yaml", STAGES / "stage4b_spread_5m.yaml", STAGES / "stage4c_spread_20m.yaml"])
    local, e2e, frm = token_budget(c)
    assert (local, e2e, str(frm)) == (0, 15_000_000, "runs/stage4b_spread_5m")
    assert steps_for_tokens(e2e, 2, 8, 1024) == 915 and c["pair"]["layers"] == "spread"
    # ridge init made for Stages 1-3 (26 layers, d_B 1024) does not fit Stage 4 (20 layers, d_B 768)
    assert _expected_init_shapes(a) == {g: (768, 1152) for g in SPREAD_MAP}
    assert _expected_init_shapes(load_config([REPO / "configs" / "base.yaml"])) == {i: (1024, 1152) for i in range(26)}


def test_layout_command(capsys):
    from transplant.cli import main

    assert main(["layout", "--config", str(REPO / "configs" / "base.yaml"),
                 "--config", str(STAGES / "stage4a_spread_1m.yaml")]) == 0
    out = capsys.readouterr().out
    assert "20 transplanted, 6 keep Gemma's FFN [2, 6, 10, 15, 19, 23]" in out
    assert "1,769,472 per layer; x 20 = 35,389,440 trainable" in out and "3<-2" in out and "25<-19" in out


def test_partial_transplant_trains_on_tiny_models(tiny_cfg):
    """Tiny stand-ins with the Stage 4 shape: A has 4 layers, B 3, layout spread keeps A layer 1's FFN."""
    build_heldout(tiny_cfg.data, tiny_tokenizer("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    tiny_cfg.pair.b.num_layers = 3
    tiny_cfg.pair.layers = "spread"
    tiny_cfg.local.enabled = False
    tiny_cfg.run.name = "stage4_tiny"
    tiny_cfg.run.tokens = 64 * 4 * 6
    run_dir = run_training(tiny_cfg, log=lambda *a: None)
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["layer_map"] == {"0": 0, "2": 1, "3": 2} and info["kept_ffn_layers"] == [1]
    proj = torch.load(run_dir / "projections.pt")
    assert sorted({k.split(".")[0] for k in proj}) and len(proj) == 6  # W_in and W_out of 3 layers
    final = json.loads((run_dir / "eval.json").read_text())
    mix = final["sets"]["mix"]
    assert all(torch.isfinite(torch.tensor(mix[k])) for k in ("ppl_a", "ppl_ablated", "ppl_hybrid"))
    assert len(final["routing"]) == 3  # one routing record per MoE layer

    from transplant.models.load import build_hybrid
    from transplant.train.checkpoint import load_projections
    from transplant.train.run import setup

    device, policy = setup(tiny_cfg)
    hybrid = build_hybrid(tiny_cfg, device, policy)
    load_projections(run_dir / "projections.pt", hybrid.model)
    layers = surgery.decoder_layers(hybrid.model)
    assert [isinstance(l.mlp, surgery.SLOT_TYPES) for l in layers] == [True, False, True, True]
    kept_before = {k: v.clone() for k, v in layers[1].mlp.state_dict().items()}
    surgery.train_mode(hybrid.model)
    ids = torch.randint(5, 50, (1, 32))
    h = hybrid.model.model(input_ids=ids).last_hidden_state
    h.float().pow(2).mean().backward()
    named = [n for n, p in hybrid.model.named_parameters() if p.grad is not None]
    assert named and all(".w_in." in n or ".w_out." in n for n in named)  # the kept FFN gets no gradient
    for k, v in layers[1].mlp.state_dict().items():
        torch.testing.assert_close(v, kept_before[k])
    # the layout floor deletes only the 3 transplanted FFNs: Gemma with layer 1's FFN in place
    with torch.no_grad(), surgery.mode(hybrid.model, "zero"):
        floor = hybrid.model(input_ids=ids).logits
    with torch.no_grad(), surgery.mode(hybrid.model, "original"):
        gemma = hybrid.model(input_ids=ids).logits
    assert not torch.allclose(floor, gemma)


def _write(d: Path, name: str, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj) if name.endswith(".json") else obj)


def _fake_run(root: Path, name: str, gap: float, steps: list, base: int, extends=None, layers=26, trainable=61_341_696,
              fact=-3.0, floor=None):
    d = root / name
    sets = {}
    for k, ce_a in (("mix", 2.33), ("wikitext103", 2.47), ("c4", 2.78)):
        sets[k] = {"ce_a": ce_a, "ce_hybrid": ce_a + gap, "ce_ablated": floor if floor is not None else 15.4,
                   "ppl_a": 10.25, "ppl_hybrid": 10.25 * 2.718281828 ** gap, "gr_ce": 0.9, "acc_a": 0.526,
                   "acc_hybrid": 0.4}
    hist = [{"step": s, "ce_a": 2.31, "ce_hybrid": 2.31 + gap + 0.5 - 0.001 * s} for s in steps]
    tokens = base + (steps[-1] + 5) * 16384 if steps else base
    _write(d, "summary.json", {"tokens_trained_total": tokens, "eval_history": hist, "final": {"sets": sets}})
    routing = {str(i): {"normalized_entropy": 0.9 - 0.01 * i, "max_load_fraction": 0.05, "dead_experts": 1}
               for i in range(layers)}
    _write(d, "eval.json", {"sets": sets, "routing": routing})
    _write(d, "run_info.json", {"trainable_params": trainable, "layer_map": {str(i): i for i in range(layers)},
                                "token_budget": {"extends": str(root / extends) if extends else None}})
    _write(d, "config.yaml", "e2e: {micro_batch: 2, grad_accum: 8}\ndata: {seq_len: 1024}\n")
    modes = {"hybrid": {"answers": ["x"] * 30, "correct": [False] * 30, "mean_fact_logprob": fact,
                        "mean_sentence_logprob": -1.0, "copies": 0},
             "gemma": {"answers": ["x"] * 30, "correct": [True] * 28 + [False] * 2, "mean_fact_logprob": -0.23,
                       "mean_sentence_logprob": -0.28, "copies": 0},
             "ablated": {"answers": ["x"] * 30, "correct": [False] * 30,
                         "mean_fact_logprob": -14.81 if floor is None else -6.0, "mean_sentence_logprob": -12.9,
                         "copies": 0}}
    _write(d, "qa.json", {"scoring": "sentence_v1", "modes": modes})
    return d


def test_stage4_report(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage4_compare", REPO / "scripts" / "stage4_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _fake_run(tmp_path, "stage1_test", 1.94, [20, 40], 245_760)
    _fake_run(tmp_path, "stage2_5m", 1.20, [50, 100, 150], 0)
    _fake_run(tmp_path, "stage3_20m", 0.83, [100, 200], 4_997_120, extends="stage2_5m")
    _fake_run(tmp_path, "stage4b_spread_5m", 1.05, [50, 100, 150], 0, layers=20, trainable=35_389_440, fact=-2.5,
              floor=4.0)
    md = mod.report(tmp_path)
    assert "| Stage 4b (`stage4b_spread_5m`) | 20 of 26 | 35.4M |" in md
    assert "**Stage 4b vs Stage 2**" in md and "| 0.82M (check, 100k tokens) |" in md
    assert "the partial transplant is better by 0.15 nats" in md
    assert "Layout floor: the 20 transplanted FFNs deleted, 6 kept" in md
    assert "| Layout floor (20 deleted, 6 kept) | 0/30 | -6.00 |" in md
    assert "| Stage 4b | 20 |" in md  # routing over 20 MoE layers
    # GR_CE of Stage 4b against the 26-deleted floor of Stages 1-3 and against its own layout floor
    gr26 = (15.4 - (2.33 + 1.05)) / (15.4 - 2.33)
    grl = (4.0 - (2.33 + 1.05)) / (4.0 - 2.33)
    assert f"| {gr26:.3f} | {grl:.3f} |" in md


def test_stage4_15m_config_and_report(tmp_path):
    cfg = _stage4_cfg("stage4_spread_15m.yaml")
    local, e2e, frm = token_budget(cfg)
    assert (local, e2e, frm) == (0, 15_000_000, None)  # from ridge init, data stream from its start
    assert steps_for_tokens(e2e, 2, 8, 1024) == 915 and cfg["pair"]["layers"] == "spread"
    assert cfg["transplant"]["init_path"] == "runs/stage4_spread_init/init.pt" and cfg["eval"]["every_steps"] == 100
    assert "15m) CONFIGS=(configs/base.yaml configs/stages/stage4_spread_15m.yaml)" in \
        (REPO / "scripts" / "stage4_run.sh").read_text()

    import importlib.util

    spec = importlib.util.spec_from_file_location("stage4_compare", REPO / "scripts" / "stage4_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _fake_run(tmp_path, "stage2_5m", 1.20, [50, 100, 150, 200, 250, 300], 0)
    _fake_run(tmp_path, "stage3_20m", 0.83, [100, 200, 300, 400, 500, 600, 700, 800, 900], 4_997_120,
              extends="stage2_5m")
    _fake_run(tmp_path, "stage4_spread_15m", 0.95, [100, 200, 300, 400, 500, 600, 700, 800, 900], 0, layers=20,
              trainable=35_389_440, fact=-2.2, floor=4.0)
    md = mod.report(tmp_path)
    assert "**Stage 4 (15M) vs all 26 layers (Stages 2 then 3, same tokens in the same order)**" in md
    assert "| 1.64M (check, 100k tokens) |" in md and "| 14.75M (check, 100k tokens) |" in md
    assert "| Stage 4 (15M) (`stage4_spread_15m`) | 20 of 26 | 35.4M |" in md
    assert "trainable parameters against 61.3M" in md
    # interpolation in log(tokens) between two checks of the all-26 curve
    assert abs(mod._interp({1e6: 2.0, 4e6: 1.0}, 2e6) - 1.5) < 1e-9 and mod._interp({1e6: 2.0}, 5e6) is None
