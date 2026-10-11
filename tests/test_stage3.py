"""Stage 3: the config, the micro-batch speed check, and that the micro-batch changes only the speed."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from transplant.config import load_config
from transplant.data.heldout import build_heldout
from transplant.models.tiny import tiny_tokenizer
from transplant.train.optim import steps_for_tokens
from transplant.train.run import run_training, token_budget

REPO = Path(__file__).resolve().parents[1]


def _speed_module():
    spec = importlib.util.spec_from_file_location("stage3_speed", REPO / "scripts" / "stage3_speed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_config_extends_stage2_to_20m_tokens():
    cfg = load_config([REPO / "configs" / "base.yaml", REPO / "configs" / "stages" / "stage2_full.yaml",
                       REPO / "configs" / "stages" / "stage2_5m.yaml", REPO / "configs" / "stages" / "stage3_20m.yaml"],
                      [])
    local, e2e, frm = token_budget(cfg)
    assert (local, e2e, str(frm)) == (0, 15_000_000, "runs/stage2_5m")
    assert cfg["run"]["name"] == "stage3_20m" and cfg["eval"]["every_steps"] == 100
    # the objective and optimizer are Stage 2's
    assert cfg["e2e"]["beta_kl"] == 1.0 and cfg["e2e"]["lr"] == 3.0e-4
    seq = int(cfg["data"]["seq_len"])
    for mb in (1, 2, 4, 8):  # every micro-batch choice gives the same steps of 16 sequences
        assert steps_for_tokens(e2e, mb, 16 // mb, seq) == 915
    assert 4_997_120 + 915 * 16 * seq == 19_988_480


def test_memcheck_end_to_end_only(tiny_cfg, tmp_path):
    from transplant.cli import memcheck

    tiny_cfg.e2e.micro_batch, tiny_cfg.e2e.grad_accum = 4, 1
    out = tmp_path / "mem.json"
    assert memcheck(tiny_cfg, 2, None, str(out), e2e_only=True) == 0
    res = json.loads(out.read_text())
    assert "local" not in res
    e = res["e2e"]
    assert e["micro_batch"] == 4 and e["grad_accum"] == 1 and e["train_tokens_per_s"] > 0


def _row(mb, peak, tps, limit=11.84):
    return {"micro_batch": mb, "grad_accum": 16 // mb, "peak_gb": peak, "limit_gb": limit, "train_tokens_per_s": tps}


def test_speed_check_picks_the_fastest_that_fits():
    sp = _speed_module()
    rows = [_row(1, 9.9, 131), _row(2, 10.3, 190), _row(4, 11.0, 240), _row(8, 11.5, 260)]
    assert sp.choose(rows, 0.75)["micro_batch"] == 4  # 8 is within 0.75 GB of the limit
    assert sp.choose([_row(1, 11.5, 131)], 0.75)["micro_batch"] == 1  # nothing fits: Stage 2's setting
    assert sp.choose([_row(1, 9.9, 131), _row(2, 10.3, 120)], 0.75)["micro_batch"] == 1  # bigger but slower


def test_speed_check_stops_before_a_micro_batch_that_would_not_fit(tmp_path, monkeypatch):
    sp = _speed_module()
    peaks = {1: 9.9, 2: 10.5, 4: 11.7, 8: 13.0}
    asked = []

    def fake_run(cmd):
        mb = int(next(c for c in cmd if c.startswith("e2e.micro_batch=")).split("=")[1])
        asked.append(mb)
        out = Path(cmd[cmd.index("--json") + 1])
        out.write_text(json.dumps({"peak_gb": peaks[mb], "limit_gb": 11.84, "process_gb": 2.0,
                                   "e2e": {"train_tokens_per_s": 100 * mb, "seconds": {}}}))

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    rows = sp.measure(tmp_path, ["a.yaml"], 3, 0.75)
    assert asked == [1, 2]  # 4 extrapolates to about 11.7 GB, over 11.09: not measured
    assert sp.choose(rows, 0.75)["micro_batch"] == 2


def test_micro_batch_changes_speed_not_training(tiny_cfg):
    """mb 2 x accum 1 and mb 1 x accum 2 read the same sequences and give the same projections."""
    build_heldout(tiny_cfg.data, tiny_tokenizer("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    tiny_cfg.local.enabled = False
    tiny_cfg.eval.every_steps = 0
    tiny_cfg.run.tokens = 64 * 2 * 4  # 4 steps of 2 sequences
    results = {}
    for mb, accum in ((2, 1), (1, 2)):
        tiny_cfg.run.name = f"mb{mb}"
        tiny_cfg.e2e.micro_batch, tiny_cfg.e2e.grad_accum = mb, accum
        run_dir = run_training(tiny_cfg, log=lambda *a: None)
        results[mb] = torch.load(run_dir / "projections.pt")
        assert json.loads((run_dir / "summary.json").read_text())["e2e_steps"] == 4
    for k in results[1]:
        torch.testing.assert_close(results[1][k], results[2][k], rtol=1e-4, atol=1e-5)
