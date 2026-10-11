#!/usr/bin/env python3
"""Progress of a training run, read from its files: safe to use while the run is going.

    python scripts/progress.py runs/stage3_20m           # once
    python scripts/progress.py runs/stage3_20m --watch   # refresh every minute (Ctrl-C stops the view only)

Run it with the project's Python (`source .venv/bin/activate` first, or `.venv/bin/python`),
not with `uv run`, so nothing is rebuilt while training runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import yaml

CHECK_MIN, FINAL_MIN = 3.0, 25.0  # minutes per progress check, and for final evaluation + samples


def _budget(cfg: dict) -> dict:
    """Steps per phase, the way train/run.py splits the run's tokens."""
    seq = int(cfg["data"]["seq_len"])
    total = int(cfg["run"]["tokens"])
    if cfg["run"].get("extend_from"):
        local = 0
    elif cfg["local"].get("enabled", True):
        local = int(round(total * float(cfg["run"].get("local_share", 0.25))))
    else:
        local = 0
    out = {}
    for phase, tokens in (("local", local), ("e2e", total - local)):
        c = cfg[phase]
        per_step = int(c["micro_batch"]) * int(c["grad_accum"]) * seq
        out[phase] = {"steps": tokens // per_step if tokens else 0, "per_step": per_step}
    return out


def _hm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def _bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _state(run: Path) -> str:
    """Running, or paused/stopped, with the last checkpoint."""
    pid = None
    try:
        pid = int((run / ".lock").read_text().strip() or 0)
        os.kill(pid, 0)
    except (OSError, ValueError):
        pid = None
    ckpt = ""
    side = run / "e2e_ckpt.json"
    if side.exists():
        try:
            c = json.loads(side.read_text())
            ckpt = f"; last checkpoint at step {c['step']}, {(time.time() - c['time']) / 60:.0f} min ago"
        except (OSError, ValueError, KeyError):
            pass
    if pid:
        return f"state: running (pid {pid}){ckpt}"
    if (run / "projections.pt").exists():
        return "state: not running; training is done, the final evaluation is not: run the same command to finish it"
    return f"state: not running (paused or stopped){ckpt}; run the same command to resume"


def report(run: Path) -> str:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    budget = _budget(cfg)
    rows = []
    if (run / "metrics.jsonl").exists():
        for x in (run / "metrics.jsonl").read_text().splitlines():
            try:
                rows.append(json.loads(x))
            except json.JSONDecodeError:  # a line cut short by a stop
                pass
    train = [r for r in rows if r.get("stage") in ("local", "e2e")]
    checks = [r for r in rows if r.get("stage") == "eval_subset"]
    lines = []
    if (run / "eval.json").exists():
        lines.append(f"{run.name}: finished. Final results in {run / 'eval.json'}, samples in {run / 'samples.md'}.")
    else:
        lines.append(_state(run))
    if not train:
        lines.append(f"{run.name}: no training step logged yet (model loading or the first step: it is slow).")
        return "\n".join(lines)
    last = train[-1]
    phase = last["stage"]
    steps = budget[phase]["steps"]
    per_step = budget[phase]["per_step"]
    done = int(last["step"]) + 1
    same = [r for r in train if r["stage"] == phase]
    gaps = [(b["time"] - a["time"]) / (b["step"] - a["step"]) for a, b in zip(same, same[1:]) if b["step"] > a["step"]]
    if gaps:  # leave out gaps that span a stop and resume
        typical = statistics.median(gaps)
        gaps = [g for g in gaps if g < 3 * typical] or gaps
    sec_per_step = statistics.median(gaps[-4:]) if gaps else None
    every = int(cfg["eval"].get("every_steps") or 0)
    checks_left = len([s for s in range(done, steps) if every and s % every == 0]) if phase == "e2e" else 0
    name = {"e2e": "end to end", "local": "local matching"}[phase]
    prev = 0
    src = cfg["run"].get("extend_from")
    if src and (Path(src) / "summary.json").exists():
        prev = int(json.loads((Path(src) / "summary.json").read_text()).get("tokens_trained_total", 0))
    total = f" ({(prev + done * per_step) / 1e6:.2f}M in total with {Path(src).name})" if prev else ""
    lines.append(f"{run.name} · {name} · step {done}/{steps} ({done / steps:.0%})  {_bar(done / steps)}  "
                 f"{done * per_step / 1e6:.2f}M / {steps * per_step / 1e6:.2f}M tokens{total}")
    age = time.time() - last["time"]
    if sec_per_step:
        left = (steps - done) * sec_per_step + 60 * (checks_left * CHECK_MIN + FINAL_MIN)
        finish = time.strftime("%a %H:%M", time.localtime(time.time() + left))
        lines.append(f"speed: {sec_per_step:.0f} s per step lately ({per_step / sec_per_step:.0f} tokens/s); "
                     f"about {_hm(done * sec_per_step)} of training so far")
        lines.append(f"left: {steps - done} steps ≈ {_hm((steps - done) * sec_per_step)}, plus {checks_left} checks and "
                     f"the final evaluation ≈ {_hm(60 * (checks_left * CHECK_MIN + FINAL_MIN))} → done around {finish}")
        if age > 3 * 10 * sec_per_step and lines[0].startswith("state: running"):
            lines.append(f"WARNING: no new line for {age / 60:.0f} min (expected one every {10 * sec_per_step / 60:.0f} min). "
                         "Is the run still going?")
    if phase == "e2e":
        lines.append(f"latest (step {done}): CE {last['ce']:.3f} · KL {last['kl']:.3f} · accuracy {last['accuracy']:.1%} · "
                     f"grad norm {last['grad_norm']:.2f} · lr {last['lr']:.2e} · peak {last.get('peak_gb', 0):.2f} GB")
    else:
        lines.append(f"latest (step {done}): L1 {last['loss']:.4f} · peak {last.get('peak_gb', 0):.2f} GB")
    if checks:
        lines.append("checks on held-out text (validation mix):")
        for c in checks:
            lines.append(f"  step {c['step']:>4}: perplexity {c['ppl_hybrid']:.1f} ({c['ppl_ratio']:.2f} × Gemma's "
                         f"{c['ppl_a']:.1f}), GR_CE {c['gr_ce']:.3f}, accuracy {c['acc_hybrid']:.1%} "
                         f"(Gemma {c['acc_a']:.1%}), routing entropy {c.get('min_routing_entropy') or 0:.3f}")
    for label, name in (("Stage 1", "stage1_test"), ("Stage 2", "stage2_5m")):
        ref = run.parent / name / "summary.json"
        if ref.exists() and run.name != name:
            mix = json.loads(ref.read_text()).get("final", {}).get("sets", {}).get("mix")
            if mix:
                lines.append(f"  {label} final: perplexity {mix['ppl_hybrid']:.1f} ({mix['ppl_ratio']:.2f} ×), "
                             f"GR_CE {mix['gr_ce']:.3f}, accuracy {mix['acc_hybrid']:.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="runs/stage3_20m")
    ap.add_argument("--watch", action="store_true", help="refresh every --every seconds")
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    run = Path(args.run)
    if not args.watch:
        print(report(run))
        return
    try:
        while True:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S  ") + report(run), flush=True)
            if (run / "eval.json").exists():
                break
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
