"""The whole pipeline on tiny models: held-out build, ridge init, local matching, end to end,
eval, samples, resume, and extending a finished run with more tokens."""
import json

import torch

from transplant.cli import main
from transplant.config import load_config
from transplant.data.heldout import build_heldout
from transplant.models.tiny import tiny_tokenizer
from transplant.train.run import run_training


def test_smoke_cli(tmp_path):
    assert main(["smoke", "--out", str(tmp_path)]) == 0
    run_dir = tmp_path / "runs" / "smoke"
    summary = json.loads((run_dir / "summary.json").read_text())
    mix = summary["final"]["sets"]["mix"]
    assert all(torch.isfinite(torch.tensor(v)) for v in (mix["ppl_hybrid"], mix["gr"], mix["acc_hybrid"]))
    assert mix["gr"] > 0.5  # the tiny transplant recovers most of the gap
    assert 0.0 <= mix["acc_hybrid"] <= 1.0 and 0.0 <= mix["acc_a"] <= 1.0
    assert (run_dir / "projections.pt").exists() and (tmp_path / "runs.csv").exists()
    assert "Hybrid:" in (run_dir / "samples.md").read_text()
    assert summary["tokens_trained_total"] <= 16384  # never more than the budget
    # Ask the trained hybrid questions, from the run folder alone.
    assert main(["ask", "--run", str(run_dir), "--max-new-tokens", "3", "--ablated"]) == 0
    asked = json.loads((run_dir / "qa.json").read_text())
    assert set(asked["modes"]) == {"gemma", "hybrid", "ablated"} and len(asked["questions"]) == 30
    assert "| Model | Correct | Fact log-prob |" in (run_dir / "qa.md").read_text()
    assert main(["ask-compare", str(run_dir), "--out", str(tmp_path / "qa_compare.md")]) == 0


def test_resume_continues_from_checkpoint(tiny_cfg):
    build_heldout(tiny_cfg.data, tiny_tokenizer("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    tiny_cfg.local.enabled = False
    run_dir = run_training(tiny_cfg, stop_after=None, log=lambda *a: None)
    ckpt = torch.load(run_dir / "e2e_ckpt.pt", weights_only=False)
    assert ckpt["step"] > 0 and "mixture" in ckpt["data_state"]
    final = torch.load(run_dir / "projections.pt")
    (run_dir / "projections.pt").unlink()
    (run_dir / "eval.json").unlink()
    run_training(tiny_cfg, resume=True, log=lambda *a: None)  # restarts from the last checkpoint
    again = torch.load(run_dir / "projections.pt")
    for k in final:
        torch.testing.assert_close(final[k], again[k], rtol=1e-5, atol=1e-6)


def test_extension_reads_new_tokens_only(tiny_cfg):
    """Stage 2 extension: continues the projections and the data stream of a finished run."""
    from transplant.data import training_stream
    from transplant.models.tiny import tiny_tokenizer as tt

    build_heldout(tiny_cfg.data, tt("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    first = run_training(tiny_cfg, log=lambda *a: None)
    end_state = torch.load(first / "e2e_data_state.pt", weights_only=False)
    tiny_cfg.run.name = "extended"
    tiny_cfg.run.extend_from = str(first)
    tiny_cfg.run.tokens = 4096
    ext = run_training(tiny_cfg, log=lambda *a: None)
    s1 = json.loads((first / "summary.json").read_text())
    s2 = json.loads((ext / "summary.json").read_text())
    assert s2["tokens_trained_total"] == s1["tokens_trained_total"] + s2["tokens_trained_this_run"]
    assert "local_steps" not in s2  # an extension is end to end only
    # The extension's stream starts exactly where the first run's stream stopped.
    a = training_stream(tiny_cfg.data, tt("a"), end_state, seed=0).next_sequence()
    fresh = training_stream(tiny_cfg.data, tt("a"), None, seed=0).next_sequence()
    assert a != fresh


def test_memory_check_runs_on_tiny_models(tiny_cfg, tmp_path):
    """`transplant memcheck` runs both phases end to end (on the CPU it reports no peak)."""
    from transplant.cli import memcheck

    out = tmp_path / "mem.json"
    assert memcheck(tiny_cfg, 1, None, str(out)) == 0
    res = json.loads(out.read_text())
    assert res["transplanted_layers"] == 4
    assert res["local"]["tokens_per_s"] > 0 and res["e2e"]["tokens_per_s"] > 0


def test_checks_before_a_resume_are_kept(tmp_path):
    from transplant.train.end_to_end import _past_checks

    rows = [{"time": 1, "stage": "e2e", "step": 0, "ce": 5.0},
            {"time": 2, "stage": "eval_subset", "step": 50, "ppl_hybrid": 64.4},
            {"time": 3, "stage": "eval_subset", "step": 100, "ppl_hybrid": 50.3},
            {"time": 4, "stage": "eval_subset", "step": 150, "ppl_hybrid": 45.0}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert _past_checks(tmp_path, 137) == [{"step": 50, "ppl_hybrid": 64.4}, {"step": 100, "ppl_hybrid": 50.3}]
    assert _past_checks(tmp_path, 0) == []
