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
  ask                 question-and-answer test, one-sentence answers: Gemma, the hybrid, No FFN (--ablated); -i to chat
  ask-compare         Gemma, No FFN and the hybrid of several runs (Stage 1, 2, 3...) side by side, from their qa.json
  layout              which Gemma layers get a transplant, which keep their FFN, trainable parameters (no model loaded)
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
    sp.add_argument("--steps", type=int, default=3, help="micro-batches per phase (the last one is timed)")
    sp.add_argument("--limit-gb", type=float, default=None,
                    help="fail above this peak (default: TRANSPLANT_MEM_LIMIT_GB, else what macOS recommends)")
    sp.add_argument("--json", default=None, help="also write the result to this file")
    sp.add_argument("--e2e-only", action="store_true", help="end-to-end steps only (no local matching)")
    sp = add("ask", "question-and-answer test with one-sentence answers (writes qa.md and qa.json in the run folder)")
    sp.add_argument("--run", default=None, help="run folder; its config.yaml and projections.pt are used")
    sp.add_argument("--projections", default=None, help="another projections file")
    sp.add_argument("--questions", default=None,
                    help="file: one question per line, as 'question | reference with [fact] | alias'")
    sp.add_argument("-i", "--interactive", action="store_true", help="type your own questions")
    sp.add_argument("--raw", action="store_true", help="continue the text as is, without the Q/A frame")
    sp.add_argument("--ablated", "--no-ffn", dest="ablated", action="store_true",
                    help="also No FFN: Gemma with all transplanted FFNs deleted")
    sp.add_argument("--only-hybrid", action="store_true", help="the trained hybrid alone: no Gemma, no No FFN")
    sp.add_argument("--max-new-tokens", type=int, default=40)
    sp = add("ask-compare", "Gemma, No FFN and the hybrid of several runs side by side (run `ask` on each first)")
    sp.add_argument("runs", nargs="+", help="run folders, earliest first, e.g. runs/stage1_test runs/stage2_5m")
    sp.add_argument("--out", default="results/qa_compare.md")
    sp = add("layout", "print the layer map and the parameter count of a config (no model is loaded)")
    sp = add("smoke", "tiny end-to-end run on CPU")
    sp.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd in ("train", "eval", "ask"):
        from .utils.progress import protect_output

        protect_output()

    if args.cmd == "smoke":
        return smoke(args.out)
    if args.cmd == "ask-compare":
        return ask_compare(args.runs, Path(args.out))
    if args.cmd == "ask" and args.run and not args.config:
        args.config = [str(Path(args.run) / "config.yaml")]  # the run's own settings

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
        from .train.interrupt import PAUSED_EXIT, StopRequested, TrainingPaused
        from .train.run import run_training

        try:
            run_training(cfg, resume=args.resume, stop_after=args.stop_after)
        except (TrainingPaused, StopRequested) as e:
            print(f"[run] {e}. Checkpoint saved; run the same command to resume.", flush=True)
            return PAUSED_EXIT
        except KeyboardInterrupt:
            print("[run] stopped at once; the last checkpoint stays. Run the same command to resume.", flush=True)
            return 130
    elif args.cmd == "eval":
        from .train.run import run_eval

        run_eval(cfg, args.projections, args.routing_agreement)
    elif args.cmd == "summarize":
        summarize(args.csv)
    elif args.cmd == "memcheck":
        return memcheck(cfg, args.steps, args.limit_gb, args.json, e2e_only=args.e2e_only)
    elif args.cmd == "ask":
        return ask_cmd(cfg, args)
    elif args.cmd == "layout":
        return layout_cmd(cfg)
    return 0


def ask_cmd(cfg, args) -> int:
    from .eval.ask import REPORT_FILE, RESULT_FILE, ask, format_markdown, interactive, load_questions
    from .models.load import build_hybrid
    from .train.checkpoint import load_projections
    from .train.run import run_dir_of, setup

    run_dir = Path(args.run) if args.run else run_dir_of(cfg)
    proj = Path(args.projections) if args.projections else run_dir / "projections.pt"
    if not proj.exists():
        print(f"[ask] no projections at {proj}; train a run first or pass --projections")
        return 1
    device, policy = setup(cfg)
    hybrid = build_hybrid(cfg, device, policy)
    load_projections(proj, hybrid.model)
    print(f"[ask] hybrid from {proj} on {device}")
    if args.only_hybrid:
        modes = ("hybrid",)
        if args.ablated:
            print("[ask] --only-hybrid: --ablated is ignored")
    else:
        modes = ("gemma", "hybrid") + (("ablated",) if args.ablated else ())
    if args.interactive:
        interactive(hybrid, device, modes=modes, max_new_tokens=max(args.max_new_tokens, 40), raw=args.raw)
        return 0
    res = ask(hybrid, load_questions(args.questions), device, modes=modes, max_new_tokens=args.max_new_tokens,
              raw=args.raw)
    md = format_markdown(res, label=f"({run_dir.name}, {proj.name})")
    print(md)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / REPORT_FILE).write_text(md)
    (run_dir / RESULT_FILE).write_text(json.dumps(res, indent=2))
    print(f"[ask] written to {run_dir / REPORT_FILE} and {RESULT_FILE}")
    return 0


def layout_cmd(cfg) -> int:
    """The surgery a config describes, from the config alone: layers, map, kept FFNs, parameters."""
    from .models.layer_map import kept_layers, resolve_layer_map
    from .models.load import b_architecture, is_tiny

    pair = cfg["pair"]
    if is_tiny(pair["a"]):
        from .models.tiny import tiny_model_a

        m = tiny_model_a()
        n_a, d_a = int(m.config.num_hidden_layers), int(m.config.hidden_size)
    else:
        n_a, d_a = int(pair["a"]["num_layers"]), int(pair["a"]["hidden_size"])
    n_b, d_b = b_architecture(pair["b"])
    a_layers, lm = resolve_layer_map(pair, n_a, n_b)
    kept = kept_layers(n_a, a_layers)
    per_layer = 2 * d_a * d_b
    print(f"A: {pair['a'].get('model_id')}  ({n_a} layers x {d_a})")
    print(f"B: {pair['b'].get('model_id')}  ({n_b} layers x {d_b}, {pair['b'].get('num_experts')} routed experts, "
          f"top-{pair['b'].get('top_k')})  revision {pair['b'].get('revision')}")
    print(f"layout: {pair.get('layers')}  ->  {len(a_layers)} transplanted, {len(kept)} keep Gemma's FFN {kept}")
    print("A layer <- B layer: " + ", ".join(f"{i}<-{j}" for i, j in lm.items()))
    print(f"projections: W_in {d_a}->{d_b} + W_out {d_b}->{d_a} = {per_layer:,} per layer; "
          f"x {len(a_layers)} = {per_layer * len(a_layers):,} trainable")
    return 0


def ask_compare(runs: list, out: Path) -> int:
    from .eval.ask import compare

    try:
        md = compare(runs)
        brief = compare(runs, full=False)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[ask-compare] {e}")
        return 1
    print(brief)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"[ask-compare] written to {out} (with every reply)")
    return 0


def memcheck(cfg, steps: int, limit_gb, json_path, e2e_only: bool = False) -> int:
    """Run memory_check and compare its peak with the limit. Exit code 1 when over."""
    import os

    from .tools import memory_check

    res = memory_check(cfg, micro_steps=steps, e2e_only=e2e_only)
    env = os.environ.get("TRANSPLANT_MEM_LIMIT_GB")
    limit = limit_gb or (float(env) if env else None) or res.get("recommended_max_gb") or 12.0
    res["limit_gb"] = round(float(limit), 2)
    measured = res["peak_gb"] == res["peak_gb"]  # NaN on the CPU: no counter there
    res["ok"] = (not measured) or res["peak_gb"] < res["limit_gb"]
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(res, indent=2))
    hours = None
    if "local" in res and res["local"]["tokens_per_s"] and res["e2e"]["tokens_per_s"]:
        hours = (5e6 / res["local"]["tokens_per_s"] + 15e6 / res["e2e"]["tokens_per_s"]) / 3600
    print(f"[memcheck] GPU peak {res['peak_gb']:.2f} GB (live tensors {res['peak_live_gb']:.2f} GB), "
          f"limit {res['limit_gb']:.2f} GB: {'OK' if res['ok'] else 'TOO HIGH'}; process (CPU side) {res['process_gb']} GB")
    e = res["e2e"]
    print(f"[memcheck] MoE forward: {res['moe_forward']}; one end-to-end micro-batch of {e['micro_batch']}: "
          f"{e['seconds']} s; training speed with {e['grad_accum']} micro-batches per step: "
          f"about {e['train_tokens_per_s']:.0f} tokens/s")
    if hours:
        print(f"[memcheck] at this speed the 20M-token Stage 2 trains in about {hours:.1f} hours "
              f"(5M tokens all end to end: {5e6 / res['e2e']['tokens_per_s'] / 3600:.1f} hours; random tokens, no evaluation)")
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
