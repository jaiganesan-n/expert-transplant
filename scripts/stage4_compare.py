#!/usr/bin/env python3
"""Stage 4 report: the partial transplant (MobileMoE-S, layout spread) against Stages 1-3.

    .venv/bin/python scripts/stage4_compare.py            # writes results/stage4_compare.md and prints it

Reads only run files (no model is loaded): summary.json, eval.json, run_info.json, config.yaml, qa.json
of runs/stage1_test, runs/stage2_5m, runs/stage3_20m and the Stage 4 runs that exist.

Two floors:
* all 26 FFNs deleted, the floor of Stages 1-3 (its losses come from their evaluation; same held-out text);
* the layout floor of Stage 4: Gemma with only the 20 transplanted FFNs deleted and the 6 kept
  (what Stage 4's own evaluation and Q/A call "ablated" / "No FFN").
Matched tokens: 4a ~ Stage 1 (1M), 4b ~ Stage 2 (5M, same tokens, checks at the same steps),
4c ~ Stage 3 (20M in total, same tokens, checks at the same steps); the 15M run ~ the all-26 curve of
Stages 2 then 3 (same tokens in the same order), interpolated at its checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

REFERENCE = [("Stage 1", "stage1_test"), ("Stage 2", "stage2_5m"), ("Stage 3", "stage3_20m")]
STAGE4 = [("Stage 4 (15M)", "stage4_spread_15m", "stages2-3"),
          ("Stage 4a", "stage4a_spread_1m", "stage1_test"), ("Stage 4b", "stage4b_spread_5m", "stage2_5m"),
          ("Stage 4c", "stage4c_spread_20m", "stage3_20m")]
SETS = ("mix", "wikitext103", "c4")
MARGIN = 0.10  # nats: the decision margin of the Stage 4 context [S4-A5]


def _load(path: Path):
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text())
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return None


class Run:
    def __init__(self, root: Path, name: str, label: str):
        d = root / name
        self.label, self.name, self.dir = label, name, d
        self.summary = _load(d / "summary.json") or {}
        self.eval = _load(d / "eval.json") or (self.summary.get("final") or {})
        self.info = _load(d / "run_info.json") or {}
        self.cfg = _load(d / "config.yaml") or {}
        self.qa = _load(d / "qa.json")
        self.finished = (d / "eval.json").exists()

    @property
    def sets(self) -> dict:
        return (self.eval or {}).get("sets", {})

    @property
    def tokens(self):
        return self.summary.get("tokens_trained_total")

    @property
    def replaced(self) -> int:
        lm = self.info.get("layer_map") or {}
        return len(lm) or 26

    @property
    def per_step(self) -> int:
        e = self.cfg.get("e2e", {})
        return int(e.get("micro_batch", 1)) * int(e.get("grad_accum", 16)) * int(self.cfg.get("data", {}).get("seq_len", 1024))

    def base_tokens(self, root: Path) -> int:
        """Tokens trained before this run's end-to-end steps: a run it extends, or its local matching."""
        src = (self.info.get("token_budget") or {}).get("extends")
        if src:
            s = _load(Path(src) / "summary.json") or _load(root / Path(src).name / "summary.json") or {}
            return int(s.get("tokens_trained_total", 0))
        return int(self.summary.get("local_tokens", 0) or 0)

    def checks(self, root: Path) -> dict:
        """{total tokens: CE gap on the 100k-token check} from the run's eval history."""
        out = {}
        base = self.base_tokens(root)
        for h in self.summary.get("eval_history", []) or []:
            if "ce_hybrid" in h and "ce_a" in h:
                out[base + int(h["step"]) * self.per_step] = h["ce_hybrid"] - h["ce_a"]
        return out


def _f(x, spec=".3f", dash="–"):
    return dash if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def _gap(s: dict):
    return None if not s else s.get("ce_hybrid", float("nan")) - s.get("ce_a", float("nan"))


def _gr(ce_floor, ce_hyb, ce_a):
    if None in (ce_floor, ce_hyb, ce_a) or ce_floor == ce_a:
        return None
    return (ce_floor - ce_hyb) / (ce_floor - ce_a)


def _mean_fact(qa: dict | None, mode: str):
    if not qa or mode not in qa.get("modes", {}):
        return None
    return qa["modes"][mode].get("mean_fact_logprob")


def _correct(qa: dict | None, mode: str):
    if not qa or mode not in qa.get("modes", {}):
        return None
    hits = [h for h in qa["modes"][mode]["correct"] if h is not None]
    return f"{sum(1 for h in hits if h)}/{len(hits)}"


def _interp(curve: dict, t: float):
    """CE gap of a learning curve {tokens: gap} at ``t`` tokens, linear in log(tokens); None outside it."""
    pts = sorted(curve.items())
    for (t0, g0), (t1, g1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            w = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0)) if t1 > t0 else 0.0
            return g0 + w * (g1 - g0)
    return next((g for tt, g in pts if tt == t), None)


def _against_curve(r: "Run", by_name: dict, root: Path, floor_layout: dict) -> list:
    """A Stage 4 run on the first N tokens of the stream, against the all-26 curve of Stages 2 and 3
    (which read the same tokens in the same order) at the same token counts."""
    s2, s3 = by_name.get("stage2_5m"), by_name.get("stage3_20m")
    if not s2 and not s3:
        return []
    curve = {}
    for ref in (s2, s3):
        if ref:
            curve.update(ref.checks(root))
    mine = r.checks(root)
    out = [f"**{r.label} vs all 26 layers (Stages 2 then 3, same tokens in the same order)**", "",
           "| Tokens | All 26, Stages 2-3 (interpolated) CE gap | " + f"{r.label} CE gap | Difference |",
           "| --- | --- | --- | --- |"]
    last = None
    for t in sorted(mine):
        g = _interp(curve, t)
        if g is None:
            continue
        out.append(f"| {t / 1e6:.2f}M (check, 100k tokens) | {g:.3f} | {mine[t]:.3f} | {mine[t] - g:+.3f} |")
        last = (t, g, mine[t])
    out += ["", "Final evaluation, 250k tokens per set (CE gap):", "",
            "| Set | " + " | ".join(f"{x.label} ({_f((x.tokens or 0) / 1e6, '.1f')}M, all 26)" for x in (s2, s3) if x)
            + f" | {r.label} ({_f((r.tokens or 0) / 1e6, '.1f')}M) |",
            "| --- | " + " | ".join("---" for x in (s2, s3) if x) + " | --- |"]
    for k in SETS:
        vals = [_gap(x.sets.get(k)) for x in (s2, s3) if x]
        out.append(f"| {k} | " + " | ".join(_f(v) for v in vals) + f" | {_f(_gap(r.sets.get(k)))} |")
    if last:
        t, g, m = last
        d = m - g
        verdict = (f"better than all 26 layers by {-d:.2f} nats" if d <= -MARGIN else
                   f"worse than all 26 layers by {d:.2f} nats" if d >= MARGIN else
                   f"within {MARGIN} nats of all 26 layers (no clear difference)")
        out += ["", f"At {t / 1e6:.2f}M tokens the partial transplant is {verdict} on the same "
                f"text (CE gap {m:.3f} vs {g:.3f}), with {_f((r.info.get('trainable_params') or 0) / 1e6, '.1f')}M "
                "trainable parameters against 61.3M. Its single learning-rate schedule differs from the two of "
                "Stages 2 and 3."]
    if "mix" in floor_layout and "mix" in r.sets:
        d = floor_layout["mix"] - r.sets["mix"]["ce_hybrid"]
        out += ["", f"Against its layout floor: {d:.2f} nats better on the mix "
                f"({'clears' if d >= MARGIN else 'does not clear'} the {MARGIN}-nat margin)."]
    return out + [""]


def report(root: Path) -> str:
    refs = [Run(root, n, label) for label, n in REFERENCE]
    s4 = [(Run(root, n, label), match) for label, n, match in STAGE4]
    refs = [r for r in refs if r.finished]
    s4 = [(r, m) for r, m in s4 if r.finished]
    by_name = {r.name: r for r in refs}
    lines = ["# Stage 4 report: Gemma 3 1B + MobileMoE-S (layout spread) against Stages 1-3", ""]
    if not s4:
        lines += ["No finished Stage 4 run yet (runs/stage4_spread_15m, stage4a_spread_1m, stage4b_spread_5m, "
                  "stage4c_spread_20m).", ""]
    # floors
    floor26 = {}
    for r in refs:
        for k in SETS:
            if k in r.sets and "ce_ablated" in r.sets[k]:
                floor26.setdefault(k, r.sets[k]["ce_ablated"])
    floor_layout = {}
    for r, _ in s4:
        for k in SETS:
            if k in r.sets and "ce_ablated" in r.sets[k]:
                floor_layout.setdefault(k, r.sets[k]["ce_ablated"])
    gemma = {}
    for r in refs + [r for r, _ in s4]:
        for k in SETS:
            if k in r.sets:
                gemma.setdefault(k, r.sets[k])

    # 1 · text
    lines += ["## 1 · Text (final evaluation, 250k held-out tokens per set)", "",
              "| Run | FFNs replaced | Trainable | Tokens | Mix ppl (× Gemma) | CE gap mix | CE gap WikiText-103 | "
              "CE gap C4 | GR_CE mix, 26-deleted floor | GR_CE mix, layout floor | Accuracy mix |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    if "mix" in gemma:
        g = gemma["mix"]
        lines.append(f"| Gemma 3 1B (target) | 0 of 26 | – | – | {g.get('ppl_a', math.exp(g['ce_a'])):.2f} (1.00) | 0 | 0 | 0 | 1 | 1 | "
                     f"{_f(g.get('acc_a'), '.1%')} |")
    for r in refs + [r for r, _ in s4]:
        m = r.sets.get("mix")
        if not m:
            continue
        gaps = [_gap(r.sets.get(k)) for k in SETS]
        is4 = r.name.startswith("stage4")
        lines.append(
            f"| {r.label} (`{r.name}`) | {r.replaced} of 26 | {_f((r.info.get('trainable_params') or 0) / 1e6, '.1f')}M | "
            f"{_f((r.tokens or 0) / 1e6, '.2f')}M | {m['ppl_hybrid']:.2f} ({m['ppl_hybrid'] / m['ppl_a']:.2f}) | "
            f"{_f(gaps[0])} | {_f(gaps[1])} | {_f(gaps[2])} | "
            f"{_f(_gr(floor26.get('mix'), m['ce_hybrid'], m['ce_a']))} | "
            f"{_f(_gr(floor_layout.get('mix'), m['ce_hybrid'], m['ce_a'])) if is4 else '–'} | {_f(m.get('acc_hybrid'), '.1%')} |")
    for label, fl in (("All 26 FFNs deleted (floor of Stages 1-3)", floor26),
                      ("Layout floor: the 20 transplanted FFNs deleted, 6 kept", floor_layout)):
        if "mix" in fl and "mix" in gemma:
            ce_a = {k: gemma[k]["ce_a"] for k in SETS if k in gemma}
            gap = {k: fl[k] - ce_a[k] for k in SETS if k in fl and k in ce_a}
            lines.append(f"| {label} | – | – | – | {math.exp(fl['mix']):,.0f} | {_f(gap.get('mix'), '.2f')} | "
                         f"{_f(gap.get('wikitext103'), '.2f')} | {_f(gap.get('c4'), '.2f')} | – | – | – |")
    lines += ["", "CE gap: hybrid minus Gemma, in nats (0 = Gemma; the headline). GR_CE against the 26-deleted floor "
              "compares with Stages 1-3; against the layout floor it is Stage 4's own.", ""]
    if "mix" in floor_layout and "mix" in floor26 and floor_layout["mix"] > floor26["mix"]:
        lines += ["Note: the layout floor is worse than deleting all 26 FFNs (the 6 kept FFNs, fed a residual stream "
                  "the 20 deleted ones no longer shape, make it worse), so GR_CE and fact recovery against it overstate "
                  "the recovery; read them against the 26-deleted floor.", ""]

    # 2 · matched tokens
    lines += ["## 2 · At matched tokens (same text, same recipe)", ""]
    for r, match in s4:
        if match == "stages2-3":
            lines += _against_curve(r, by_name, root, floor_layout)
            continue
        ref = by_name.get(match)
        if not ref:
            continue
        lines += [f"**{r.label} vs {ref.label}**", "",
                  "| Tokens | " + f"{ref.label} CE gap | {r.label} CE gap | Difference |", "| --- | --- | --- | --- |"]
        a, b = ref.checks(root), r.checks(root)
        for t in sorted(set(a) & set(b)):
            lines.append(f"| {t / 1e6:.2f}M (check, 100k tokens) | {a[t]:.3f} | {b[t]:.3f} | {b[t] - a[t]:+.3f} |")
        for k in SETS:
            ga, gb = _gap(ref.sets.get(k)), _gap(r.sets.get(k))
            if ga is not None and gb is not None:
                lines.append(f"| final, {k} ({_f((ref.tokens or 0) / 1e6, '.2f')}M vs {_f((r.tokens or 0) / 1e6, '.2f')}M) | "
                             f"{ga:.3f} | {gb:.3f} | {gb - ga:+.3f} |")
        gm_a, gm_b = _gap(ref.sets.get("mix")), _gap(r.sets.get("mix"))
        if gm_a is not None and gm_b is not None:
            d = gm_b - gm_a
            verdict = (f"better by {-d:.2f} nats (at least {MARGIN})" if d <= -MARGIN else
                       f"worse by {d:.2f} nats" if d >= MARGIN else f"within {MARGIN} nats of it (no clear difference)")
            note = " (Stage 1 also spent 0.25M tokens on local matching)" if match == "stage1_test" else ""
            lines += ["", f"Mix CE gap: {r.label} {gm_b:.3f} vs {ref.label} {gm_a:.3f}: the partial transplant is "
                      f"{verdict}{note}, with {_f((r.info.get('trainable_params') or 0) / 1e6, '.1f')}M trainable "
                      f"parameters against {_f((ref.info.get('trainable_params') or 0) / 1e6, '.1f')}M.", ""]
        if "mix" in floor_layout and "mix" in r.sets:
            d = floor_layout["mix"] - r.sets["mix"]["ce_hybrid"]
            lines += [f"Against its layout floor: {d:.2f} nats better on the mix "
                      f"({'clears' if d >= MARGIN else 'does not clear'} the {MARGIN}-nat margin).", ""]

    # 3 · Q/A
    lines += ["## 3 · Q/A test (30 questions, one-sentence answers)", "",
              "| Model | Correct | Fact log-prob | Fact recovery (26 deleted = 0%) | Fact recovery (layout floor = 0%) | "
              "Sentence log-prob | Copied |", "| --- | --- | --- | --- | --- | --- | --- |"]
    q_g = next((_mean_fact(r.qa, "gemma") for r in refs + [r for r, _ in s4] if _mean_fact(r.qa, "gemma") is not None), None)
    q_26 = next((_mean_fact(r.qa, "ablated") for r in refs if _mean_fact(r.qa, "ablated") is not None), None)
    q_lay = next((_mean_fact(r.qa, "ablated") for r, _ in s4 if _mean_fact(r.qa, "ablated") is not None), None)

    def rec(x, floor):
        return None if None in (x, floor, q_g) or q_g == floor else (x - floor) / (q_g - floor)

    src_g = next((r.qa for r in refs + [r for r, _ in s4] if _mean_fact(r.qa, "gemma") is not None), None)
    if src_g:
        lines.append(f"| Gemma 3 1B | {_correct(src_g, 'gemma')} | {_f(q_g, '.2f')} | 100% | 100% | "
                     f"{_f(src_g['modes']['gemma'].get('mean_sentence_logprob'), '.2f')} | – |")
    for r in refs + [r for r, _ in s4]:
        if not r.qa or "hybrid" not in r.qa.get("modes", {}):
            continue
        h = r.qa["modes"]["hybrid"]
        x = h.get("mean_fact_logprob")
        is4 = r.name.startswith("stage4")
        lines.append(f"| {r.label} hybrid | {_correct(r.qa, 'hybrid')} | {_f(x, '.2f')} | {_f(rec(x, q_26), '.0%')} | "
                     f"{_f(rec(x, q_lay), '.0%') if is4 else '–'} | {_f(h.get('mean_sentence_logprob'), '.2f')} | "
                     f"{h.get('copies', '–')}/{len(h.get('answers', []))} |")
    if q_26 is not None:
        src = next(r.qa for r in refs if _mean_fact(r.qa, "ablated") is not None)
        lines.append(f"| All 26 FFNs deleted | {_correct(src, 'ablated')} | {_f(q_26, '.2f')} | 0% | – | "
                     f"{_f(src['modes']['ablated'].get('mean_sentence_logprob'), '.2f')} | – |")
    if q_lay is not None:
        src = next(r.qa for r, _ in s4 if _mean_fact(r.qa, "ablated") is not None)
        lines.append(f"| Layout floor (20 deleted, 6 kept) | {_correct(src, 'ablated')} | {_f(q_lay, '.2f')} | "
                     f"{_f(rec(q_lay, q_26), '.0%')} | 0% | {_f(src['modes']['ablated'].get('mean_sentence_logprob'), '.2f')} | – |")
    lines += ["", "Fact log-prob: mean log-probability per token of the fact after the reference sentence's opening. "
              "Run `bash scripts/stage4_validate.sh <run>` for a Stage 4 run without qa.json.", ""]

    # 4 · routing
    lines += ["## 4 · Routing (final evaluation)", "",
              "| Run | MoE layers | Lowest entropy (layer) | Mean entropy | Dead experts | Busiest expert (layer) | "
              "Agreement with Model B's own routing |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in refs + [r for r, _ in s4]:
        rt = (r.eval or {}).get("routing") or {}
        if not rt:
            continue
        ent = {k: v["normalized_entropy"] for k, v in rt.items()}
        low = min(ent, key=ent.get)
        busy = max(rt, key=lambda k: rt[k]["max_load_fraction"])
        dead = sum(v["dead_experts"] for v in rt.values())
        full = _load(r.dir / "eval_full.json") or {}
        agr = full.get("agreement") or {}
        js = [v["jaccard"] for v in agr.values() if isinstance(v, dict) and "jaccard" in v]
        mean_j = sum(js) / len(js) if js else None
        n_exp = int(((r.cfg.get("pair") or {}).get("b") or {}).get("num_experts") or 60)
        lines.append(f"| {r.label} | {len(rt)} | {ent[low]:.3f} ({low}) | {sum(ent.values()) / len(ent):.3f} | "
                     f"{dead} of {len(rt) * n_exp} | {rt[busy]['max_load_fraction']:.1%} ({busy}) | "
                     f"{_f(mean_j, '.3f')} |")
    lines += ["", "Entropy alarm: 0.7. Layer numbers are Gemma's. Agreement (mean Jaccard of the top-4 sets, chance "
              "0.039) appears after `bash scripts/stage4_validate.sh <run> --agreement`.", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="the runs folder")
    ap.add_argument("--out", default="results/stage4_compare.md")
    args = ap.parse_args()
    md = report(Path(args.runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"[stage4] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
