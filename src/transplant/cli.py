"""Command line: ``python -m transplant <command> --config FILE [--config FILE ...] [--set key=value ...]``.

Commands, in the order a project runs them:

  inspect         M0  module names in Gemma and MobileMoE (pin them in configs/pairs/primary.yaml)
  data-check      M0  stream a few documents from every data source
  build-heldout   M0  validation, calibration and shift sets (never trained on)
  align-stats     M0  share of token positions the two tokenizers align on
  overflow-check      optional: Gemma in fp16 vs fp32 (the M5 runs bf16, so not needed)
  baselines           optional: Gemma, Gemma with all FFNs zeroed, bits per byte (train computes its own)
  train               ridge init -> local matching -> end to end -> evaluation -> samples (resumable)
  eval                re-evaluate saved projections and write samples (optionally routing agreement)
  summarize           results/runs.csv, one line per run and evaluation set
  memcheck            memory and speed of real training steps on this machine (fresh process)
  smoke               the whole pipeline on tiny stand-in models, on CPU, in about a minute
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import load_config

REPO = Path(__file__).resolve().parents[2]


def _cfg(args):
    paths = args.config or [str(REPO / "configs" / "base.yaml")]
    return load_config(paths, args.set or [])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="transplant", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", action="append", help="YAML config (repeatable; later wins)")
        sp.add_argument("--set", action="append", help="override, e.g. e2e.lr=1e-4 (repeatable)")
        return sp

    add("inspect", "M0: print and record module names").add_argument("--out", default="runs/m0")
    sp = add("data-check", "M0: read a few documents from every source")
    sp.add_argument("--docs", type=int, default=3)
    sp.add_argument("--include-optional", action="store_true")
    sp = add("build-heldout", "build held-out validation/calibration/shift sets")
    sp.add_argument("--skip-shift", action="store_true")
    sp = add("overflow-check", "M0: Gemma fp16 vs fp32")
    sp.add_argument("--seqs", type=int, default=100)
    sp.add_argument("--out", default="runs/m0")
    sp = add("align-stats", "M0: tokenizer alignment share")
    sp.add_argument("--chars", type=int, default=1_000_000)
    sp.add_argument("--out", default="runs/m0")
    sp = add("baselines", "optional reference numbers for Gemma")
    sp.add_argument("--layer-sets", default="all", help="semicolon-separated layer specs to ablate")
    sp.add_argument("--out", default="runs/baselines")
    sp = add("train", "train one configuration")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--stop-after", choices=["init", "local"], default=None)
    sp = add("eval", "evaluate saved projections")
    sp.add_argument("--projections", default=None)
    sp.add_argument("--routing-agreement", action="store_true")
    sp = add("summarize", "aggregate results/runs.csv over seeds")
    sp.add_argument("--csv", default="results/runs.csv")
    sp = add("memcheck", "memory and speed of one local-matching and one end-to-end step, real models")
    sp.add_argument("--steps", type=int, default=2, help="micro-batches per phase (the last one is timed)")
    sp.add_argument("--limit-gb", type=float, default=None,
                    help="fail above this peak (default: TRANSPLANT_MEM_LIMIT_GB, else what macOS recommends)")
    sp.add_argument("--json", default=None, help="also write the result to this file")
    sp = add("smoke", "tiny end-to-end run on CPU")
    sp.add_argument("--out", default=None)

    args = p.parse_args(argv)

    if args.cmd == "smoke":
        return smoke(args.out)

    cfg = _cfg(args)
    if args.cmd == "inspect":
        from .tools import inspect_models

        inspect_models(cfg, Path(args.out))
    elif args.cmd == "data-check":
        from .tools import data_check

        rows = data_check(cfg, args.docs, args.include_optional)
        bad = [r["name"] for r in rows if not r["ok"]]
        if bad:
            print(f"[data] unavailable: {bad}")
            return 1
    elif args.cmd == "build-heldout":
        from .data.heldout import build_heldout, build_shift_set
        from .models.load import load_tokenizer

        tok = load_tokenizer(cfg["pair"]["a"], "a")
        build_heldout(cfg["data"], tok, cfg["data"]["heldout_dir"])
        if not args.skip_shift:
            for spec in cfg["data"].get("shift_sets", []):
                build_shift_set(spec, tok, cfg["data"]["heldout_dir"], int(cfg["data"]["seq_len"]))
    elif args.cmd == "overflow-check":
        from .tools import overflow_check
        from .utils.device import pick_device

        overflow_check(cfg, pick_device(cfg.get("device", "auto")), args.seqs, Path(args.out))
    elif args.cmd == "align-stats":
        from .tools import align_stats

        align_stats(cfg, args.chars, Path(args.out))
    elif args.cmd == "baselines":
        from .eval.baselines import run_baselines
        from .train.run import setup

        device, policy = setup(cfg)
        run_baselines(cfg, device, policy, [s.strip() for s in args.layer_sets.split(";") if s.strip()], Path(args.out))
    elif args.cmd == "train":
        from .train.run import run_training

        run_training(cfg, resume=args.resume, stop_after=args.stop_after)
    elif args.cmd == "eval":
        from .train.run import run_eval

        run_eval(cfg, args.projections, args.routing_agreement)
    elif args.cmd == "summarize":
        summarize(args.csv)
    elif args.cmd == "memcheck":
        return memcheck(cfg, args.steps, args.limit_gb, args.json)
    return 0


def memcheck(cfg, steps: int, limit_gb, json_path) -> int:
    """Run memory_check and compare its peak with the limit. Exit code 1 when over."""
    import os

    from .tools import memory_check

    res = memory_check(cfg, micro_steps=steps)
    env = os.environ.get("TRANSPLANT_MEM_LIMIT_GB")
    limit = limit_gb or (float(env) if env else None) or res.get("recommended_max_gb") or 12.0
    res["limit_gb"] = round(float(limit), 2)
    measured = res["peak_gb"] == res["peak_gb"]  # NaN on the CPU: no counter there
    res["ok"] = (not measured) or res["peak_gb"] < res["limit_gb"]
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(res, indent=2))
    hours = None
    if res["local"]["tokens_per_s"] and res["e2e"]["tokens_per_s"]:
        hours = (5e6 / res["local"]["tokens_per_s"] + 15e6 / res["e2e"]["tokens_per_s"]) / 3600
    print(f"[memcheck] peak {res['peak_gb']:.2f} GB (live tensors {res['peak_live_gb']:.2f} GB), "
          f"limit {res['limit_gb']:.2f} GB: {'OK' if res['ok'] else 'TOO HIGH'}")
    if hours:
        print(f"[memcheck] at this speed Stage 2 trains in about {hours:.1f} hours (random tokens, no evaluation)")
    print(json.dumps(res))
    return 0 if res["ok"] else 1


def summarize(csv_path: str) -> None:
    import csv
    import statistics
    from collections import defaultdict

    latest = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):  # a re-evaluated run keeps only its latest row
            latest[(row["run_name"], row["metric_set"])] = row
    groups = defaultdict(list)
    for row in latest.values():
        key = (row["experiment"], row["run_name"].rsplit("_s", 1)[0], row["metric_set"])
        groups[key].append(row)

    def col(rows, k, fmt="{:.3f}"):
        vals = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        if len(vals) == 1:
            return fmt.format(mean)
        return f"{fmt.format(mean)} +- {fmt.format(statistics.stdev(vals))}"

    print("stage | run | set | tokens | ppl A | ppl hybrid | GR_CE | GR | acc A | acc hybrid | min routing entropy")
    for (exp, run, metric_set), rows in sorted(groups.items()):
        print(" | ".join([exp, run, metric_set, col(rows, "tokens", "{:,.0f}"), col(rows, "ppl_a"), col(rows, "ppl_hybrid"),
                          col(rows, "gr_ce"), col(rows, "gr"), col(rows, "acc_a"), col(rows, "acc_hybrid"),
                          col(rows, "min_routing_entropy")]))


def smoke(out: str | None) -> int:
    """Everything end to end on tiny stand-in models and synthetic text."""
    from .data.heldout import build_heldout
    from .models.load import load_tokenizer
    from .train.run import run_eval, run_training

    root = Path(out) if out else Path(tempfile.mkdtemp(prefix="transplant_smoke_"))
    overrides = [
        f"run.out_dir={root / 'runs'}",
        f"data.heldout_dir={root / 'heldout'}",
        f"log.runs_csv={root / 'runs.csv'}",
    ]
    cfg = load_config([REPO / "configs" / "base.yaml", REPO / "configs" / "pairs" / "tiny.yaml"], overrides)
    tok = load_tokenizer(cfg["pair"]["a"], "a")
    build_heldout(cfg["data"], tok, cfg["data"]["heldout_dir"])
    run_dir = run_training(cfg)
    res = run_eval(cfg, routing_agreement=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    # Then add tokens the way a Stage 2 extension does: same stream, no token read twice.
    ext = load_config([REPO / "configs" / "base.yaml", REPO / "configs" / "pairs" / "tiny.yaml"],
                      overrides + ["run.name=smoke_extended", f"run.extend_from={run_dir}", "run.tokens=4096"])
    ext_dir = run_training(ext)
    ext_summary = json.loads((ext_dir / "summary.json").read_text())
    print(f"[smoke] done: {run_dir}")
    print(f"[smoke] final: {json.dumps(summary['final']['sets'], default=float)[:600]}")
    print(f"[smoke] routing agreement: {json.dumps(res.get('agreement', {}), default=float)[:300]}")
    print(f"[smoke] extension: {ext_summary['tokens_trained_total']:,} tokens trained in total; "
          f"samples in {ext_dir / 'samples.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
