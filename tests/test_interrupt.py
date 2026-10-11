"""Pausing and interruptions: a stopped run resumes exactly where it stopped and ends with the same
projections as a run that was never stopped."""
import json
import os
import signal
import time

import pytest
import torch

from transplant.data.heldout import build_heldout
from transplant.models.tiny import tiny_tokenizer
from transplant.train.checkpoint import load_latest_checkpoint, save_checkpoint
from transplant.train.interrupt import RunLock, StopGuard, StopRequested, TrainingPaused, lock_holder
from transplant.train.run import run_training

STEPS = 8


def _setup(cfg, every=0):
    build_heldout(cfg.data, tiny_tokenizer("a"), cfg.data.heldout_dir, log=lambda *a: None)
    cfg.local.enabled = False
    cfg.eval.every_steps = every
    cfg.e2e.micro_batch, cfg.e2e.grad_accum = 2, 2
    cfg.run.tokens = STEPS * 2 * 2 * int(cfg.data.seq_len)
    return cfg


def _reference(cfg):
    cfg.run.name = "never_stopped"
    run_dir = run_training(cfg, log=lambda *a: None)
    return torch.load(run_dir / "projections.pt")


def _same(a, b):
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5)


def _sigint():
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.05)  # the handler runs here, in the main thread


# ---- the guard and the lock ---------------------------------------------------------------------

def test_guard_stops_at_once_but_never_inside_a_critical_block(tmp_path):
    with StopGuard(tmp_path / "PAUSE", log=lambda m: None) as g:
        with pytest.raises(StopRequested):
            _sigint()
        g.reason, g.presses = None, 0
        finished = False
        with pytest.raises(StopRequested):
            with g.critical():
                _sigint()
                finished = True  # the block still ran to its end
        assert finished
        g.reason, g.presses = None, 0
        (tmp_path / "PAUSE").touch()
        with pytest.raises(StopRequested):
            g.check()
        g.clear_pause_file()
        assert not (tmp_path / "PAUSE").exists()
        g.reason, g.presses = None, 2
        with pytest.raises(KeyboardInterrupt):  # the third Ctrl-C quits at once
            with g.shielded():
                _sigint()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # handlers restored


def test_run_lock(tmp_path):
    (tmp_path / ".lock").write_text(str(os.getppid()))  # a live process holds it
    with pytest.raises(RuntimeError, match="still working"):
        RunLock(tmp_path).acquire()
    (tmp_path / ".lock").write_text("999999")  # a process that is gone: taken over
    with RunLock(tmp_path):
        assert lock_holder(tmp_path) == os.getpid()
    assert not (tmp_path / ".lock").exists() and lock_holder(tmp_path) is None


def test_checkpoint_falls_back_to_the_previous_one(tiny_cfg, cpu, fp32_policy, tmp_path):
    from transplant.models.load import build_hybrid
    from transplant.train.optim import make_optimizer, make_scheduler

    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    from transplant.models import surgery

    params = surgery.trainable_parameters(hybrid.model)
    opt = make_optimizer(params, 1e-3, 0.0, tiny_cfg.optim)
    sched = make_scheduler(opt, "cosine", 10, 1, 0.1)
    path = tmp_path / "e2e_ckpt.pt"
    for step in (1, 2):
        save_checkpoint(path, model=hybrid.model, optimizer=opt, scheduler=sched, scaler=None, step=step,
                        data_state={"at": step})
    assert json.loads((tmp_path / "e2e_ckpt.json").read_text())["step"] == 2
    assert load_latest_checkpoint(path, model=hybrid.model, optimizer=opt, scheduler=sched)["step"] == 2
    path.write_bytes(b"cut short by a power cut")
    payload = load_latest_checkpoint(path, model=hybrid.model, optimizer=opt, scheduler=sched, log=lambda m: None)
    assert payload["step"] == 1 and payload["data_state"] == {"at": 1}


# ---- training ------------------------------------------------------------------------------------

def test_pause_file_then_resume(tiny_cfg):
    from transplant.train.run import run_dir_of

    cfg = _setup(tiny_cfg)
    ref = _reference(cfg)
    cfg.run.name = "paused"
    run_dir = run_dir_of(cfg)

    def log(msg):
        if str(msg).startswith("[e2e] step 3/"):
            (run_dir / "PAUSE").touch()

    with pytest.raises(TrainingPaused) as info:
        run_training(cfg, log=log)
    assert info.value.step == 3 and info.value.reason == "pause file"
    assert torch.load(run_dir / "e2e_ckpt.pt", weights_only=False)["step"] == 3
    assert not (run_dir / "PAUSE").exists() and not (run_dir / "eval.json").exists()
    assert lock_holder(run_dir) is None
    run_training(cfg, resume=True, log=lambda *a: None)
    _same(ref, torch.load(run_dir / "projections.pt"))


def test_ctrl_c_then_resume(tiny_cfg):
    cfg = _setup(tiny_cfg)
    ref = _reference(cfg)
    cfg.run.name = "ctrl_c"

    def log(msg):
        if str(msg).startswith("[e2e] step 5/"):
            _sigint()

    with pytest.raises(TrainingPaused) as info:
        run_training(cfg, log=log)
    assert info.value.step == 5 and info.value.reason == "SIGINT"
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    run_dir = run_training(cfg, resume=True, log=lambda *a: None)
    _same(ref, torch.load(run_dir / "projections.pt"))


def test_a_check_cut_short_runs_again(tiny_cfg):
    cfg = _setup(tiny_cfg, every=4)
    ref = _reference(cfg)
    cfg.run.name = "check_cut"
    fired = []

    def log(msg):
        if str(msg).startswith("[eval] mix:") and not fired:
            fired.append(1)
            _sigint()

    with pytest.raises(TrainingPaused) as info:
        run_training(cfg, log=log)
    assert info.value.step == 4
    seen = []
    run_dir = run_training(cfg, resume=True, log=seen.append)
    assert any("cut short" in str(m) for m in seen)
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r["stage"] == "eval_subset"] == [4]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert [h["step"] for h in summary["eval_history"]] == [4]
    _same(ref, torch.load(run_dir / "projections.pt"))


def test_an_error_saves_then_resumes(tiny_cfg, monkeypatch):
    from transplant.data.pack import PackedStream

    cfg = _setup(tiny_cfg)
    ref = _reference(cfg)
    cfg.run.name = "network_drop"
    real = PackedStream.next_batch
    calls = {"n": 0}

    def flaky(self, n):
        calls["n"] += 1
        if calls["n"] == 8:  # step 3, its second micro-batch
            raise ConnectionError("network down")
        return real(self, n)

    monkeypatch.setattr(PackedStream, "next_batch", flaky)
    with pytest.raises(ConnectionError):
        run_training(cfg, log=lambda *a: None)
    from transplant.train.run import run_dir_of

    run_dir = run_dir_of(cfg)
    assert torch.load(run_dir / "e2e_ckpt.pt", weights_only=False)["step"] == 3
    assert lock_holder(run_dir) is None
    monkeypatch.setattr(PackedStream, "next_batch", real)
    run_training(cfg, resume=True, log=lambda *a: None)
    _same(ref, torch.load(run_dir / "projections.pt"))


def test_final_evaluation_keeps_finished_sets(tiny_cfg):
    cfg = _setup(tiny_cfg)
    cfg.run.name = "final_cut"

    def log(msg):
        if str(msg).startswith("[eval] mix:"):
            _sigint()

    with pytest.raises(TrainingPaused) as info:
        run_training(cfg, log=log)
    assert info.value.phase == "final evaluation"
    from transplant.train.run import run_dir_of

    run_dir = run_dir_of(cfg)
    assert (run_dir / "projections.pt").exists() and not (run_dir / "eval.json").exists()
    partial = json.loads((run_dir / "eval_partial.json").read_text())
    assert "mix" in partial["results"]["sets"]
    seen = []
    run_training(cfg, resume=True, log=seen.append)
    assert any("already evaluated before the stop: mix" in str(m) for m in seen)
    final = json.loads((run_dir / "eval.json").read_text())
    assert final["sets"]["mix"]["ppl_hybrid"] == pytest.approx(partial["results"]["sets"]["mix"]["ppl_hybrid"])
    assert final["routing"] and not (run_dir / "eval_partial.json").exists()
    assert json.loads((run_dir / "summary.json").read_text())["tokens_trained_total"] > 0


def test_cli_returns_the_pause_code(monkeypatch, tmp_path):
    import transplant.train.run as run_mod
    from transplant.cli import main
    from transplant.train.interrupt import PAUSED_EXIT

    def paused(*a, **k):
        raise TrainingPaused(3, 8, "SIGINT")

    monkeypatch.setattr(run_mod, "run_training", paused)
    assert main(["train", "--set", f"run.out_dir={tmp_path}"]) == PAUSED_EXIT
