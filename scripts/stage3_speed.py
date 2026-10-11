#!/usr/bin/env python3
"""Pick the micro-batch for a long run by measuring it on this Mac: memory and speed of real steps.

For micro-batch 1, 2, 4 and 8 (gradient accumulation 16, 8, 4 and 2: always 16 sequences, 16,384
tokens, per optimizer step, so the training is the same and only the speed changes), it runs
`transplant memcheck --e2e-only` in a fresh process. A larger micro-batch gives the GPU bigger
matrix products per kernel (each of the 60 experts gets more tokens), which is where the time
goes on the Apple GPU. It stops before a micro-batch whose memory, extrapolated from the ones
measured, would pass the limit, and keeps the fastest one whose peak stays MARGIN_GB below the
limit macOS recommends for the GPU. Writes <run>/speed.json (and each measurement in <run>/speed/).

    uv run python scripts/stage3_speed.py runs/stage3_20m \\
        --config runs/stage2_5m/config.yaml --config configs/stages/stage3_20m.yaml

scripts/stage3_run.sh runs it once, before training; after a stop, finished measurements are kept.
Delete <run>/speed.json and <run>/speed/ to measure again.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

SEQS_PER_STEP = 16
CANDIDATES = (1, 2, 4, 8)
MARGIN_GB = 0.75  # training also runs evaluations and keeps more kernels cached than a short check


def _fits(row: dict, margin: float) -> bool:
    peak = row["peak_gb"]
    return isinstance(peak, (int, float)) and (math.isnan(peak) or peak <= row["limit_gb"] - margin)


def measure(run: Path, configs: list[str], steps: int, margin: float) -> list[dict]:
    out_dir = run / "speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for mb in CANDIDATES:
        if len(rows) >= 2 and not any(math.isnan(r["peak_gb"]) for r in rows[-2:]):
            a, b = rows[-2], rows[-1]
            slope = (b["peak_gb"] - a["peak_gb"]) / (b["micro_batch"] - a["micro_batch"])
            predicted = b["peak_gb"] + slope * (mb - b["micro_batch"])
            if predicted > b["limit_gb"] - margin:
                print(f"[speed] micro-batch {mb}: about {predicted:.1f} GB expected, over "
                      f"{b['limit_gb'] - margin:.1f} GB; not measured", flush=True)
                break
        accum = SEQS_PER_STEP // mb
        out = out_dir / f"mb{mb}.json"
        r = None
        if out.exists():  # measured before a stop: kept (delete the speed folder to measure again)
            try:
                r = json.loads(out.read_text())
                print(f"[speed] micro-batch {mb}: measured before, reusing {out}", flush=True)
            except (OSError, json.JSONDecodeError):
                r = None
        if r is None:
            cmd = [sys.executable, "-m", "transplant", "memcheck", "--e2e-only", "--steps", str(steps),
                   "--json", str(out)]
            for c in configs:
                cmd += ["--config", c]
            cmd += ["--set", f"e2e.micro_batch={mb}", "--set", f"e2e.grad_accum={accum}"]
            print(f"[speed] micro-batch {mb} x {accum} per step: measuring (fresh process, about 2 to 4 min)", flush=True)
            rc = subprocess.run(cmd).returncode
            if rc in (-2, 130):  # Ctrl-C
                raise KeyboardInterrupt
            if not out.exists():
                print(f"[speed] micro-batch {mb} failed (exit code {rc}); larger ones are not tried", flush=True)
                break
            r = json.loads(out.read_text())
        e = r["e2e"]
        row = {"micro_batch": mb, "grad_accum": accum, "train_tokens_per_s": e["train_tokens_per_s"],
               "peak_gb": r["peak_gb"], "limit_gb": r["limit_gb"], "process_gb": r.get("process_gb"),
               "seconds": e["seconds"]}
        rows.append(row)
        print(f"[speed] micro-batch {mb}: about {row['train_tokens_per_s']:.0f} tokens/s, GPU peak "
              f"{row['peak_gb']:.2f} GB (limit {row['limit_gb']:.2f}, kept below {row['limit_gb'] - margin:.2f})",
              flush=True)
        if not _fits(row, margin):
            print(f"[speed] micro-batch {mb} is too close to the memory limit; larger ones are not tried", flush=True)
            break
    return rows


def choose(rows: list[dict], margin: float) -> dict:
    fits = [r for r in rows if _fits(r, margin)]
    if fits:
        best = max(fits, key=lambda r: r["train_tokens_per_s"])
        return {**best, "why": "fastest micro-batch that stays below the memory limit with margin"}
    # nothing measured fits with margin: Stage 2's setting, which trained at a 9.8 GB peak
    return {"micro_batch": 1, "grad_accum": SEQS_PER_STEP, "train_tokens_per_s": None, "peak_gb": None,
            "why": "no measured micro-batch stayed below the limit with margin; Stage 2's setting"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run folder, e.g. runs/stage3_20m")
    ap.add_argument("--config", action="append", required=True, help="the run's configs, as for `train`")
    ap.add_argument("--steps", type=int, default=3, help="micro-batches per measurement (the first compiles)")
    ap.add_argument("--margin-gb", type=float, default=MARGIN_GB)
    args = ap.parse_args()
    run = Path(args.run)
    rows = measure(run, args.config, args.steps, args.margin_gb)
    pick = choose(rows, args.margin_gb)
    result = {"micro_batch": pick["micro_batch"], "grad_accum": pick["grad_accum"],
              "train_tokens_per_s": pick["train_tokens_per_s"], "peak_gb": pick["peak_gb"], "why": pick["why"],
              "margin_gb": args.margin_gb, "candidates": rows}
    (run / "speed.json").write_text(json.dumps(result, indent=2))

    print("\n| Micro-batch | Accumulation | Training speed | GPU peak |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        mark = "  <- chosen" if r["micro_batch"] == result["micro_batch"] else ""
        print(f"| {r['micro_batch']} | {r['grad_accum']} | {r['train_tokens_per_s']:.0f} tokens/s | "
              f"{r['peak_gb']:.2f} GB |{mark}")
    try:  # the run's token budget, for a time estimate
        from transplant.config import load_config

        cfg = load_config(args.config, [])
        seq = int(cfg["data"]["seq_len"])
        steps = int(cfg["run"]["tokens"]) // (SEQS_PER_STEP * seq)
        tps = result["train_tokens_per_s"]
        if tps:
            hours = steps * SEQS_PER_STEP * seq / tps / 3600
            checks = steps // int(cfg["eval"].get("every_steps") or steps + 1)
            print(f"\n[speed] chosen: micro-batch {result['micro_batch']} x {result['grad_accum']} ({result['why']}). "
                  f"{steps} steps at about {tps:.0f} tokens/s: about {hours:.1f} hours of training, plus about "
                  f"{(checks * 3 + 30) / 60:.1f} hours of evaluation. Short measurements; a long run on a fanless "
                  "Mac can be somewhat slower once it heats up.")
        else:
            print(f"\n[speed] chosen: micro-batch 1 x 16 ({result['why']})")
    except Exception as e:  # the estimate is only informative
        print(f"[speed] chosen: micro-batch {result['micro_batch']} x {result['grad_accum']} (no time estimate: {e})")
    print(f"[speed] written to {run / 'speed.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
