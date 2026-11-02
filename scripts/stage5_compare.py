#!/usr/bin/env python3
"""Stage 5 report: did MobileMoE-S's pretrained experts contribute? Random experts against Stage 4.

    .venv/bin/python scripts/stage5_compare.py            # writes results/stage5_compare.md and prints it

Reads only run files (no model is loaded): runs/stage4_spread_15m (MobileMoE-S's pretrained experts),
runs/stage5_random_15m (the same blocks with every weight random, frozen), their ridge inits
(runs/stage4_spread_init, runs/stage5_random_init) and, for the floor with all 26 FFNs deleted, Stages 1-3.

The two runs differ only in the weights inside the 20 MoE blocks: same layout, same 35.4M trained
projections, same W_in at the start, the same 15M tokens in the same order, the same schedule, the same
evaluation. So the difference between them is what the pretrained experts' weights add:

    contribution = CE gap (random experts) - CE gap (pretrained experts)        (nats; > 0: they helped)

Decision margin: 0.10 nats, as in Stage 4.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_compare import REFERENCE, SETS, Run, _correct, _f, _gap, _gr, _load, _mean_fact  # noqa: E402

PRETRAINED = ("Stage 4, pretrained experts", "stage4_spread_15m", "stage4_spread_init")
RANDOM = ("Stage 5, random experts", "stage5_random_15m", "stage5_random_init")
MARGIN = 0.10  # nats, as in Stage 4


def sign_test(wins: int, n: int) -> float:
    """Two-sided exact binomial p-value of ``wins`` out of ``n`` under a fair coin."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def verdict(delta: float) -> str:
    if delta >= MARGIN:
        return (f"**The pretrained experts contributed.** With random experts of the same architecture the hybrid "
                f"is {delta:.2f} nats worse on the mix ({math.exp(delta):.2f} × the perplexity), with the same "
                "trained parameters, data, recipe and starting W_in. Part of Stage 4's recovery comes from what "
                "MobileMoE-S's experts learned in pre-training.")
    if delta <= -MARGIN:
        return (f"**Random experts did better**, by {-delta:.2f} nats on the mix. MobileMoE-S's pretrained weights "
                "did not help this hybrid; the recovery comes from Gemma and the trained projections around a frozen "
                "nonlinear block.")
    return (f"**No clear contribution from the pretrained experts**: random experts of the same architecture come "
            f"within {MARGIN} nats of them on the mix ({delta:+.3f}). The recovery comes mainly from Gemma (its "
            "attention, embeddings and 6 kept FFNs) and the trained projections around a frozen nonlinear block, "
            "not from MobileMoE-S's pre-trained knowledge.")


def _ridge_layers(path: Path) -> dict:
    rep = _load(path) or {}
    return {int(k): v for k, v in (rep.get("layers") or {}).items()}


def _nmse(layer: dict, which: str):
    v = (layer or {}).get(which) or {}
    return v.get("val_nmse")


def _routing_row(label: str, r: Run) -> str | None:
    rt = (r.eval or {}).get("routing") or {}
    if not rt:
        return None
    ent = {k: v["normalized_entropy"] for k, v in rt.items()}
    low = min(ent, key=ent.get)
    busy = max(rt, key=lambda k: rt[k]["max_load_fraction"])
    dead = sum(v["dead_experts"] for v in rt.values())
    n_exp = int(((r.cfg.get("pair") or {}).get("b") or {}).get("num_experts") or 60)
    return (f"| {label} | {ent[low]:.3f} ({low}) | {sum(ent.values()) / len(ent):.3f} | {dead} of {len(rt) * n_exp} | "
            f"{rt[busy]['max_load_fraction']:.1%} ({busy}) |")


def report(root: Path) -> str:
    pre, rnd = Run(root, PRETRAINED[1], PRETRAINED[0]), Run(root, RANDOM[1], RANDOM[0])
    refs = [r for r in (Run(root, n, label) for label, n in REFERENCE) if r.finished]
    lines = ["# Stage 5 report: random experts against MobileMoE-S's pretrained experts (Stage 4)", ""]
    if not pre.finished:
        return "\n".join(lines + [f"Stage 4 has not finished (`{pre.dir}/eval.json` is missing).", ""]) + "\n"
    if not rnd.finished:
        lines += [f"Stage 5 has not finished yet (`{rnd.dir}/eval.json` is missing): `bash scripts/stage5_run.sh` "
                  "trains and evaluates it. Checks made so far are compared in section 2.", ""]

    floor26 = next((r.sets["mix"]["ce_ablated"] for r in refs if "mix" in r.sets), None)
    gaps = {k: (_gap(pre.sets.get(k)), _gap(rnd.sets.get(k))) for k in SETS}

    # Verdict
    if rnd.finished and None not in gaps["mix"]:
        delta = gaps["mix"][1] - gaps["mix"][0]
        lines += ["## Verdict", "", verdict(delta), ""]
        others = [f"{k} {gaps[k][1] - gaps[k][0]:+.3f}" for k in SETS[1:] if None not in gaps[k]]
        if others:
            lines += [f"Contribution on text unlike the training mix (random minus pretrained, CE gap in nats): "
                      f"{', '.join(others)}.", ""]
        qp, qr = pre.qa, rnd.qa
        fp = ((qp or {}).get("modes", {}).get("hybrid") or {}).get("fact_logprob")
        fr = ((qr or {}).get("modes", {}).get("hybrid") or {}).get("fact_logprob")
        if fp and fr:
            wins = sum(1 for a, b in zip(fp, fr) if a > b)
            ties = sum(1 for a, b in zip(fp, fr) if a == b)
            n = len(fp) - ties
            lines += [f"Facts: the right fact is more likely with the pretrained experts for {wins} of {n} questions "
                      f"(sign test p = {sign_test(wins, n):.3g}); mean fact log-prob "
                      f"{_mean_fact(qp, 'hybrid'):.2f} (pretrained) vs {_mean_fact(qr, 'hybrid'):.2f} (random); "
                      f"correct {_correct(qp, 'hybrid')} vs {_correct(qr, 'hybrid')}.", ""]

    # 1 · text
    lines += ["## 1 · Text (final evaluation, 250k held-out tokens per set)", "",
              "| Model | Mix ppl (× Gemma) | CE gap mix | CE gap WikiText-103 | CE gap C4 | Accuracy mix | "
              "GR_CE mix (26 deleted = 0) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    g = pre.sets.get("mix")
    if g:
        lines.append(f"| Gemma 3 1B | {g['ppl_a']:.2f} (1.00) | 0 | 0 | 0 | {_f(g.get('acc_a'), '.1%')} | 1 |")
    for r in (pre, rnd):
        m = r.sets.get("mix")
        if not m:
            continue
        lines.append(f"| {r.label} (`{r.name}`) | {m['ppl_hybrid']:.2f} ({m['ppl_hybrid'] / m['ppl_a']:.2f}) | "
                     + " | ".join(_f(_gap(r.sets.get(k))) for k in SETS)
                     + f" | {_f(m.get('acc_hybrid'), '.1%')} | {_f(_gr(floor26, m['ce_hybrid'], m['ce_a']))} |")
    if rnd.finished and pre.sets.get("mix") and rnd.sets.get("mix"):
        mp, mr = pre.sets["mix"], rnd.sets["mix"]
        lines.append(f"| **Random minus pretrained** | × {mr['ppl_hybrid'] / mp['ppl_hybrid']:.2f} | "
                     + " | ".join(_f(None if None in gaps[k] else gaps[k][1] - gaps[k][0], "+.3f") for k in SETS)
                     + f" | {_f(mr.get('acc_hybrid', 0) - mp.get('acc_hybrid', 0), '+.1%')} | – |")
    if "mix" in pre.sets:
        lay = pre.sets["mix"]["ce_ablated"] - pre.sets["mix"]["ce_a"]
        lines.append(f"| Layout floor (the 20 FFNs deleted, no block at all) | {math.exp(pre.sets['mix']['ce_ablated']):,.0f} | "
                     f"{lay:.2f} | " + " | ".join(_f(pre.sets[k]["ce_ablated"] - pre.sets[k]["ce_a"], ".2f")
                                                    for k in SETS[1:] if k in pre.sets) + " | – | – |")
    lines += ["", "CE gap: hybrid minus Gemma, in nats (0 = Gemma). Both runs: 20 of 26 FFNs replaced, 35.4M trained "
              "parameters, 15M tokens.", ""]

    # 2 · learning curve
    a, b = pre.checks(root), rnd.checks(root)
    common = sorted(set(a) & set(b))
    lines += ["## 2 · Learning curve: the same tokens, in the same order, with the same schedule", ""]
    if common:
        lines += ["| Tokens | Pretrained experts, CE gap | Random experts, CE gap | Random minus pretrained |",
                  "| --- | --- | --- | --- |"]
        lines += [f"| {t / 1e6:.2f}M (check, 100k tokens) | {a[t]:.3f} | {b[t]:.3f} | {b[t] - a[t]:+.3f} |"
                  for t in common]
        lines += ["", "Both runs check at the same steps of the same cosine schedule, so every row compares the two at "
                  "exactly the same point of training.", ""]
    else:
        lines += ["No check of Stage 5 yet.", ""]

    # 3 · ridge init
    lp, lr = _ridge_layers(root / PRETRAINED[2] / "init.json"), _ridge_layers(root / RANDOM[2] / "init.json")
    both = sorted(set(lp) & set(lr))
    lines += ["## 3 · Before training: ridge W_out through each kind of block", ""]
    if both:
        lines += ["W_in is the same in both runs (Stage 4's ridge fit). W_out is fitted by ridge to Gemma's own FFN "
                  "output through the block in the slot; validation NMSE (0 = exact, 1 = no better than zero):", "",
                  "| Gemma layer | W_in val NMSE (shared) | W_out val NMSE, pretrained block | W_out val NMSE, random block |",
                  "| --- | --- | --- | --- |"]
        rows = [(i, _nmse(lp[i], "w_in"), _nmse(lp[i], "w_out"), _nmse(lr[i], "w_out")) for i in both]
        lines += [f"| {i} | {_f(wi)} | {_f(wp)} | {_f(wr)} |" for i, wi, wp, wr in rows]
        mp = [wp for _, _, wp, _ in rows if wp is not None]
        mr = [wr for _, _, _, wr in rows if wr is not None]
        if mp and mr:
            lines += [f"| **mean** | – | **{sum(mp) / len(mp):.3f}** | **{sum(mr) / len(mr):.3f}** |"]
        lines += [""]
    else:
        lines += ["Ridge init reports missing (`runs/stage4_spread_init/init.json`, `runs/stage5_random_init/init.json`).", ""]

    # 4 · Q/A
    lines += ["## 4 · Q/A test (30 questions, one-sentence answers)", "",
              "| Model | Correct | Fact log-prob | Fact recovery (26 deleted = 0%) | Sentence log-prob | Copied |",
              "| --- | --- | --- | --- | --- | --- |"]
    q_g = _mean_fact(pre.qa, "gemma")
    q_26 = next((_mean_fact(r.qa, "ablated") for r in refs if _mean_fact(r.qa, "ablated") is not None), None)

    def rec(x):
        return None if None in (x, q_26, q_g) or q_g == q_26 else (x - q_26) / (q_g - q_26)

    if q_g is not None:
        lines.append(f"| Gemma 3 1B | {_correct(pre.qa, 'gemma')} | {q_g:.2f} | 100% | "
                     f"{_f(pre.qa['modes']['gemma'].get('mean_sentence_logprob'), '.2f')} | – |")
    for r in (pre, rnd):
        if not r.qa or "hybrid" not in r.qa.get("modes", {}):
            continue
        h = r.qa["modes"]["hybrid"]
        lines.append(f"| {r.label} | {_correct(r.qa, 'hybrid')} | {_f(h.get('mean_fact_logprob'), '.2f')} | "
                     f"{_f(rec(h.get('mean_fact_logprob')), '.0%')} | {_f(h.get('mean_sentence_logprob'), '.2f')} | "
                     f"{h.get('copies', '–')}/{len(h.get('answers', []))} |")
    if q_26 is not None:
        lines.append(f"| All 26 FFNs deleted | 0/30 | {q_26:.2f} | 0% | – | – |")
    if pre.qa and rnd.qa and "hybrid" in rnd.qa.get("modes", {}):
        cp, cr = pre.qa["modes"]["hybrid"]["correct"], rnd.qa["modes"]["hybrid"]["correct"]
        qs = pre.qa.get("questions") or [f"question {i + 1}" for i in range(len(cp))]
        only_p = [qs[i] for i, (x, y) in enumerate(zip(cp, cr)) if x and not y]
        only_r = [qs[i] for i, (x, y) in enumerate(zip(cp, cr)) if y and not x]
        n_both = sum(1 for x, y in zip(cp, cr) if x and y)
        lines += ["", f"Right with both: {n_both}. Only with the pretrained experts ({len(only_p)}): "
                  f"{'; '.join(only_p) or '–'}. Only with random experts ({len(only_r)}): {'; '.join(only_r) or '–'}."]
    lines += ["", "Every reply: `runs/stage4_spread_15m/qa.md` and `runs/stage5_random_15m/qa.md`.", ""]

    # 5 · routing
    lines += ["## 5 · Routing (final evaluation, 20 MoE layers)", "",
              "| Experts | Lowest entropy (Gemma layer) | Mean entropy | Dead experts | Busiest expert (layer) |",
              "| --- | --- | --- | --- | --- |"]
    lines += [row for row in (_routing_row(r.label, r) for r in (pre, rnd)) if row]
    lines += ["", "Entropy alarm: 0.7. W_in alone decides which experts a token reaches, in both runs.", ""]

    # 6 · cost and what was random
    lines += ["## 6 · Runs", "", "| Run | Trainable | Tokens | Training speed | Peak memory |", "| --- | --- | --- | --- | --- |"]
    for r in (pre, rnd):
        if r.summary:
            lines.append(f"| {r.label} | {_f((r.info.get('trainable_params') or 0) / 1e6, '.1f')}M | "
                         f"{_f((r.tokens or 0) / 1e6, '.2f')}M | {_f(r.summary.get('e2e_tokens_per_s'), '.0f')} tok/s | "
                         f"{_f(r.summary.get('peak_gb'), '.1f')} GB |")
    drawn = rnd.info.get("random_moe")
    if drawn:
        lines += ["", f"Random draw of Stage 5: seed {drawn.get('seed')}, full {drawn.get('full')}, fingerprint "
                  f"`{drawn.get('fingerprint')}` (`scripts/verify_stage5_random.py {rnd.dir}` checks it)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="the runs folder")
    ap.add_argument("--out", default="results/stage5_compare.md")
    args = ap.parse_args()
    md = report(Path(args.runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"[stage5] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
