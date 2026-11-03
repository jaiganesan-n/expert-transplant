"""Stage 5 result charts (PNG, 4,000 px wide) into docs/figures/results/, drawn from the run files.

    uv run --extra figures python docs/figures/src/make_stage5_figures.py      # needs runs/ (git-ignored)

stage5_matched_tokens.png  CE gap at every check: pretrained against random experts, all 26 layers beside them
stage5_final_eval.png      CE gap and next-token accuracy on the three held-out sets
qa_stage4_5.png            fact log-probability per question, pretrained against random experts
routing_stage4_5.png       routing entropy per layer at the end, and the weakest layer at every check

Same look as the charts of Stages 1-4 (Stage 3 rust, Stage 4 purple, Gemma blue); Stage 5, the random
experts, is amber. Labels are in ink, not in the series colour: amber is too light to read as text.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import BG, FAINT, INK, MUTED, plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = HERE.parents[2]
RUNS = ROOT / "runs"
OUT = HERE.parent / "results"

S3, S4, S5, GEMMA = "#B4441A", "#4A3AA7", "#EDA100", "#2A78D6"
ALARM, KEPT_BAND = "#D03B3B", "#DCE6F2"
GRID, AXIS, TICK, RULE, TEXT2 = "#E6E5E0", "#C8C7BE", "#52514E", "#D0D3D8", "#333333"

TOKENS_PER_STEP = 16_384
STAGE2_TOKENS = 4_997_120  # Stage 3 continues Stage 2 from here
KEPT = [2, 6, 10, 15, 19, 23]
SETS = [("mix", "Validation mix"), ("wikitext103", "WikiText-103"), ("c4", "C4 validation")]
ALARM_ENTROPY = 0.7

# Short labels for the 30 questions of `transplant ask`, in its order; the answer comes from qa.json.
SHORT = ["Capital of France", "Capital of Japan", "Capital of Germany", "Pyramids of Giza", "Language of Brazil",
         "Longest river in Africa", "Number of continents", "Gas plants absorb", "Formula of water",
         "Freezing point of water", "The Red Planet", "Largest planet", "Star at the center",
         "Organ that pumps blood", "Force toward the Earth", "2 + 2", "Square root of 81", "Sides of a triangle",
         "10 times 10", "Days in a week", "Months in a year", "Day after Monday", "Colour of the sky",
         "Opposite of hot", "King of the jungle", "Main ingredient of guacamole", "Author of Romeo and Juliet",
         "First US president", "First person on the Moon", "Year World War II ended"]
TOPIC_NAMES = {"geography": "geography", "science": "science", "history": "history", "everyday": "everyday",
               "math": "arithmetic"}


def load(name: str) -> SimpleNamespace:
    d = RUNS / name
    rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line.strip()]
    qa = d / "qa.json"
    return SimpleNamespace(checks=[r for r in rows if "ce_hybrid" in r],
                           eval=json.loads((d / "eval.json").read_text()),
                           qa=json.loads(qa.read_text()) if qa.exists() else None)


def gap(row: dict) -> float:
    return row["ce_hybrid"] - row["ce_a"]


def mtok(step: int, offset: int = 0) -> float:
    return (offset + step * TOKENS_PER_STEP) / 1e6


def sign_test(wins: int, n: int) -> float:
    tail = sum(math.comb(n, k) for k in range(max(wins, n - wins), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------------
# Layout helpers: positions in inches from the top-left corner, as in the Stage 1-4 charts.
def figure(w: float, h: float, title: str, subtitle: str):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=BG)
    fig.text(0.4 / w, 1 - 0.36 / h, title, fontsize=18, fontweight="bold", color=INK, va="center")
    fig.add_artist(Line2D([0.4 / w, 1 - 0.4 / w], [1 - 0.75 / h] * 2, color=RULE, lw=1.2, transform=fig.transFigure))
    fig.text(0.4 / w, 1 - 1.1 / h, subtitle, fontsize=11.5, color=MUTED, va="center")
    return fig


def axes(fig, left: float, top: float, width: float, height: float, title: str | None = None):
    w, h = fig.get_size_inches()
    ax = fig.add_axes([left / w, 1 - (top + height) / h, width / w, height / h])
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TICK, labelsize=10.5, length=3, color=AXIS)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    if title:
        fig.text(left / w, 1 - (top - 0.26) / h, title.upper(), fontsize=12.5, fontweight="bold", color=MUTED,
                 va="center")
    return ax


def label_axes(ax, x: str, y: str) -> None:
    ax.set_xlabel(x, fontsize=11.5, color=TICK, labelpad=8)
    ax.set_ylabel(y, fontsize=11.5, color=TICK, labelpad=8)


def legend(fig, handles, y_in: float, **kw) -> None:
    w, h = fig.get_size_inches()
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.4 / w, y_in / h), ncol=len(handles),
               frameon=False, fontsize=11, labelcolor=TEXT2, **{"handlelength": 2.2, "columnspacing": 3.2, **kw})


def footnote(fig, text: str, y_in: float = 0.28, x_in: float = 0.4, ha: str = "left") -> None:
    w, h = fig.get_size_inches()
    fig.text(x_in / w, y_in / h, text, fontsize=10, color=FAINT, ha=ha, va="center")


def line(ax, x, y, color, marker="o", ms=6.5, zorder=3):
    ax.plot(x, y, color=color, lw=2.2, marker=marker, ms=ms, mec=BG, mew=1.4, zorder=zorder)


def line_handle(color, label, marker="o"):
    return Line2D([], [], color=color, lw=2.2, marker=marker, ms=7, mec=BG, mew=1.2, label=label)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


# ----------------------------------------------------------------------------------
def matched_tokens(pre, rnd, s2, s3) -> None:
    fig = figure(20, 8, "Same tokens, same order: MobileMoE-S's pretrained experts against random ones",
                 "CE gap to Gemma on the validation mix at every check (100k held-out tokens). Stage 5 is Stage 4 "
                 "with every expert weight random and frozen; nothing else differs.")
    ax = axes(fig, 1.0, 1.85, 13.2, 4.5)
    all26 = [(mtok(r["step"]), gap(r)) for r in s2.checks] + \
            [(mtok(r["step"], STAGE2_TOKENS), gap(r)) for r in s3.checks]
    line(ax, *zip(*all26), S3, ms=5.5, zorder=2)
    line(ax, [mtok(r["step"]) for r in pre.checks], [gap(r) for r in pre.checks], S4, marker="D", ms=6.5)
    line(ax, [mtok(r["step"]) for r in rnd.checks], [gap(r) for r in rnd.checks], S5)
    ax.axvline(STAGE2_TOKENS / 1e6, color=AXIS, lw=0.9, zorder=1)
    ax.text(STAGE2_TOKENS / 1e6 + 0.15, 1.9, "Stage 2 ends; Stage 3 restarts\nits learning-rate schedule",
            fontsize=10, color=MUTED, va="top")

    xe, ye = all26[-1]
    ax.text(xe + 0.3, ye, "all 26 (M), pretrained", fontsize=11, fontweight="bold", color=INK, va="center")
    p_end, r_end = pre.checks[-1], rnd.checks[-1]
    x = mtok(p_end["step"])
    ax.text(x + 0.3, gap(r_end) + 0.03, "Stage 5, random experts", fontsize=11, fontweight="bold", color=INK,
            va="bottom")
    ax.text(x + 0.3, gap(p_end) - 0.03, "Stage 4, pretrained experts", fontsize=11, fontweight="bold", color=INK,
            va="top")
    # bracket: random minus pretrained at the last check, between the two end labels
    bx = x + 0.4
    ax.plot([bx, bx], [gap(p_end) + 0.02, gap(r_end) - 0.02], color=INK, lw=1.2)
    for yy in (gap(p_end) + 0.02, gap(r_end) - 0.02):
        ax.plot([bx - 0.12, bx], [yy, yy], color=INK, lw=1.2)
    ax.text(bx + 0.2, (gap(p_end) + gap(r_end)) / 2 + 0.04, f"+{gap(r_end) - gap(p_end):.2f}", fontsize=12,
            fontweight="bold", color=INK, ha="left", va="center")
    ax.set_xlim(0, 22.5)
    ax.set_ylim(0.7, 1.95)
    label_axes(ax, "tokens trained (millions)", "CE gap to Gemma (nats, lower is better)")

    # the difference at every check
    w, h = fig.get_size_inches()
    fig.text(15.2 / w, 1 - 1.62 / h, "RANDOM MINUS PRETRAINED", fontsize=12.5, fontweight="bold", color=MUTED,
             va="center")
    fig.text(15.2 / w, 1 - 1.98 / h, "at the same tokens (nats; positive = pretrained better)", fontsize=10.5,
             color=MUTED, va="center")
    bx_ax = axes(fig, 16.0, 2.3, 3.3, 3.9)
    diffs = [(mtok(p["step"]), gap(r) - gap(p)) for p, r in zip(pre.checks, rnd.checks)]
    ys = list(range(len(diffs)))[::-1]
    bx_ax.barh(ys, [d for _, d in diffs], height=0.5, color=S5)
    for yy, (_, d) in zip(ys, diffs):
        bx_ax.text(d + 0.012, yy, f"+{d:.2f}", fontsize=11, fontweight="bold", color=INK, va="center")
    bx_ax.set_yticks(ys, [f"{t:.2f}M" for t, _ in diffs], fontsize=11, family="DejaVu Sans Mono")
    bx_ax.set_xlim(0, 0.62)
    bx_ax.xaxis.set_visible(False)
    bx_ax.yaxis.grid(False)
    for side in ("left", "bottom"):
        bx_ax.spines[side].set_visible(False)
    bx_ax.tick_params(length=0, pad=12)

    legend(fig, [line_handle(S3, "Stages 2-3, all 26 layers (MobileMoE-M, pretrained)"),
                 line_handle(S4, "Stage 4, 20 of 26 (MobileMoE-S, pretrained)", marker="D"),
                 line_handle(S5, "Stage 5, 20 of 26 (MobileMoE-S, random)")], 0.78)
    footnote(fig, "Stages 4 and 5 read the same 15M tokens in the same order, with the same schedule and the "
                  "same starting W_in; both check at the same steps. The difference stops shrinking after 5M tokens.")
    save(fig, "stage5_matched_tokens.png")


# ----------------------------------------------------------------------------------
def final_eval(pre, rnd) -> None:
    fig = figure(20, 8.2, "Final evaluation: pretrained against random experts, 15M tokens each",
                 "250k held-out tokens per set, the same sequences for every model. Stage 5 = Stage 4 with random, "
                 "frozen experts of the same architecture.")
    ce = axes(fig, 1.0, 1.88, 8.3, 4.95, "CE gap to Gemma")
    acc = axes(fig, 11.3, 1.88, 8.3, 4.95, "Next-token accuracy")
    for i, (key, _) in enumerate(SETS):
        p, r = pre.eval["sets"][key], rnd.eval["sets"][key]
        gp, gr = gap(p), gap(r)
        for x, v, c in ((i - 0.19, gp, S4), (i + 0.19, gr, S5)):
            ce.bar(x, v, width=0.36, color=c)
            ce.text(x, v + 0.03, f"{v:.2f}", fontsize=11, fontweight="bold", color=INK, ha="center", va="bottom")
        top = max(gp, gr) + 0.2
        ce.plot([i - 0.19, i - 0.19, i + 0.19, i + 0.19], [top - 0.04, top, top, top - 0.04], color=INK, lw=1.1)
        ce.text(i, top + 0.03, f"+{gr - gp:.2f}", fontsize=12, fontweight="bold", color=INK, ha="center",
                va="bottom")
        for x, v, c in ((i - 0.28, p["acc_hybrid"], S4), (i, r["acc_hybrid"], S5), (i + 0.28, p["acc_a"], GEMMA)):
            acc.bar(x, 100 * v, width=0.26, color=c)
            acc.text(x, 100 * v + 0.8, f"{100 * v:.1f}", fontsize=11, fontweight="bold", color=INK, ha="center",
                     va="bottom")
    for ax in (ce, acc):
        ax.set_xticks(range(len(SETS)), [name for _, name in SETS], fontsize=11.5)
        ax.tick_params(axis="x", length=0, pad=6)
    ce.set_ylim(0, 2.2)
    acc.set_ylim(0, 60)
    ce.set_ylabel("CE gap to Gemma (nats, lower is better)", fontsize=11.5, color=TICK, labelpad=8)
    acc.set_ylabel("next-token accuracy (%)", fontsize=11.5, color=TICK, labelpad=8)
    legend(fig, [Patch(color=S4, label="Stage 4 (15M, 20 of 26, MobileMoE-S, pretrained experts)"),
                 Patch(color=S5, label="Stage 5 (15M, 20 of 26, MobileMoE-S, random experts)"),
                 Patch(color=GEMMA, label="Gemma 3 1B")], 0.85, handlelength=1.4)
    footnote(fig, "Brackets: random minus pretrained, in nats. The verdict margin, set before the run, was "
                  "0.10 nats on the validation mix.")
    save(fig, "stage5_final_eval.png")


# ----------------------------------------------------------------------------------
def qa(pre, rnd, s3) -> None:
    hp, hr, g = pre.qa["modes"]["hybrid"], rnd.qa["modes"]["hybrid"], pre.qa["modes"]["gemma"]
    floor = s3.qa["modes"]["ablated"]["mean_fact_logprob"]  # all 26 FFNs deleted
    expected, topics = pre.qa["expected"], pre.qa["topics"]
    assert len(SHORT) == len(expected) == len(hp["fact_logprob"]) == len(hr["fact_logprob"])
    n = len(SHORT)
    wins = sum(a > b for a, b in zip(hp["fact_logprob"], hr["fact_logprob"]))

    fig = figure(20, 15.5, "Does the fact come back? Pretrained against random experts on the 30 questions",
                 "Mean log-probability per token of the right fact after the reference opening (“The capital "
                 "of France is” → “Paris”); 0 = certain. ✓ = the reply contained the answer.")
    ax = axes(fig, 5.0, 1.78, 9.4, 12.4)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    ys = list(range(n))[::-1]
    for y, a, b, c in zip(ys, hp["fact_logprob"], hr["fact_logprob"], g["fact_logprob"]):
        ax.plot([a, b], [y, y], color=RULE, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(b, y, "o", color=S5, ms=9, mec=BG, mew=1.5, zorder=3)
        ax.plot(a, y, "D", color=S4, ms=8.5, mec=BG, mew=1.5, zorder=4)
        ax.plot(c, y, "o", color=GEMMA, ms=7, mec=BG, mew=1.2, zorder=2)
    cols = {"S4": (0.75, hp["correct"]), "S5": (1.45, hr["correct"])}
    for name, (x, correct) in cols.items():
        ax.text(x, n - 0.3, name, fontsize=11, fontweight="bold", color=INK, ha="center", va="bottom")
        for y, ok in zip(ys, correct):
            ax.text(x, y, "✓" if ok else "·", fontsize=13 if ok else 11, color=INK if ok else FAINT,
                    ha="center", va="center")
    lo = math.floor(min(hp["fact_logprob"] + hr["fact_logprob"])) - 0.5
    ax.set_xlim(lo, 2.0)
    ax.set_xticks(range(math.ceil(lo), 1))
    ax.set_ylim(-0.7, n + 0.2)
    ax.set_yticks(ys, [f"{s}  ({e[0]})" for s, e in zip(SHORT, expected)], fontsize=11, color=TEXT2)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("fact log-probability per token (higher is better)", fontsize=11.5, color=TICK, labelpad=8)

    # means, and what they say
    w, h = fig.get_size_inches()
    side = fig.add_axes([15.2 / w, 0.1, 4.5 / w, 1 - 0.1 - 1.9 / h])
    side.set_axis_off()
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.text(0, 1, "MEANS OVER THE 30 QUESTIONS", fontsize=12.5, fontweight="bold", color=MUTED, va="top")
    for x, head in ((0.52, "fact\nlog-prob"), (0.74, "fact\nrecovery"), (0.93, "correct")):
        side.text(x, 0.955, head, fontsize=10, color=MUTED, ha="center", va="top")

    def recovery(v: float) -> str:
        return f"{100 * (v - floor) / (g['mean_fact_logprob'] - floor):.0f}%"

    rows = [("D", S4, "Stage 4, pretrained", hp), ("o", S5, "Stage 5, random", hr), ("o", GEMMA, "Gemma 3 1B", g)]
    for k, (marker, color, name, mode) in enumerate(rows):
        y = 0.89 - 0.045 * k
        side.plot(0.015, y, marker, color=color, ms=9, mec=BG, mew=1.2)
        side.text(0.05, y, name, fontsize=11, color=TEXT2, va="center")
        side.text(0.52, y, f"{mode['mean_fact_logprob']:.2f}", fontsize=11, fontweight="bold", color=INK,
                  ha="center", va="center")
        side.text(0.74, y, recovery(mode["mean_fact_logprob"]), fontsize=11, fontweight="bold", color=INK,
                  ha="center", va="center")
        side.text(0.93, y, f"{sum(mode['correct'])}/{n}", fontsize=11, color=INK, ha="center", va="center")

    by_topic = {}
    for t, a, b in zip(topics, hp["fact_logprob"], hr["fact_logprob"]):
        by_topic.setdefault(t, []).append(a - b)
    only_pre = sum(a and not b for a, b in zip(hp["correct"], hr["correct"]))
    only_rnd = sum(b and not a for a, b in zip(hp["correct"], hr["correct"]))
    both = sum(a and b for a, b in zip(hp["correct"], hr["correct"]))
    p = sign_test(wins, n)
    mant, exp = f"{p:.0e}".split("e")
    exception = [s for s, a, b in zip(SHORT, hp["fact_logprob"], hr["fact_logprob"]) if a <= b]
    notes = (f"Fact recovery: between “all 26 FFNs\ndeleted” (0%) and Gemma (100%).\n\n"
             f"The fact is more likely with the\npretrained experts for {wins} of {n}\nquestions "
             f"(sign test p = {mant} × 10$^{{{int(exp)}}}$)"
             + (f";\nthe exception: {', '.join(e.lower() for e in exception)}." if exception else "."))
    style = dict(fontsize=11, color=TEXT2, va="top", linespacing=1.55)
    side.text(0, 0.76, notes, **style)
    side.text(0, 0.6, "Pretrained minus random, by topic\n(fact log-prob, nats per token):", **style)
    ranked = sorted(by_topic.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    for k, (t, v) in enumerate(ranked):
        y = 0.55 - 0.024 * k
        side.text(0.04, y, TOPIC_NAMES.get(t, t), **style)
        side.text(0.42, y, f"+{sum(v) / len(v):.1f}", **{**style, "fontweight": "bold", "color": INK}, ha="right")
    side.text(0, 0.55 - 0.024 * len(ranked) - 0.02,
              f"Right with both: {both}. Only with the\npretrained experts: {only_pre}; only with\n"
              f"random experts: {only_rnd}.", **style)

    footnote(fig, "Grey bar: the range between Stage 5 and Stage 4 for that question. Data: runs/stage4_spread_15m/"
                  "qa.json and runs/stage5_random_15m/qa.json; the all-26-deleted floor from runs/stage3_20m/qa.json.")
    save(fig, "qa_stage4_5.png")


# ----------------------------------------------------------------------------------
def routing(pre, rnd) -> None:
    fig = figure(20, 8.2, "Routing health: every layer at the end, and the weakest layer over training",
                 "Normalized expert-load entropy (1 = all 60 experts used equally; alarm at 0.7). Gemma's layer "
                 "numbers. Blue bands: the 6 layers that keep Gemma's FFN.")
    ax = axes(fig, 1.0, 1.88, 11.0, 4.7, "Final evaluation, per layer")
    right = axes(fig, 13.2, 1.88, 6.3, 4.7, "The weakest layer, check by check")
    for k in KEPT:
        ax.axvspan(k - 0.45, k + 0.45, color=KEPT_BAND, lw=0, zorder=0)
    ax.text(KEPT[0], 0.993, "kept\nGemma FFN", fontsize=9.5, color=MUTED, ha="center", va="top")

    summary = []
    for run, color, marker, name in ((pre, S4, "D", "Stage 4"), (rnd, S5, "o", "Stage 5")):
        r = run.eval["routing"]
        layers = sorted(int(k) for k in r)
        ent = [r[str(l)]["normalized_entropy"] for l in layers]
        line(ax, layers, ent, color, marker=marker, ms=6 if marker == "D" else 6.5)
        lo = min(layers, key=lambda l: r[str(l)]["normalized_entropy"])
        summary.append((name, lo, r[str(lo)], sum(v["dead_experts"] for v in r.values()), 60 * len(r),
                        sum(ent) / len(ent)))
        steps = [mtok(c["step"]) for c in run.checks]
        mins = [c["min_routing_entropy"] for c in run.checks]
        line(right, steps, mins, color, marker=marker, ms=6 if marker == "D" else 6.5)
        above_alarm = mins[-1] > ALARM_ENTROPY  # keep the label clear of the alarm line
        right.text(steps[-1] + 0.5, mins[-1] + (0.03 if above_alarm else 0), f"{name}\n{mins[-1]:.3f}",
                   fontsize=11, fontweight="bold", color=INK, va="center")

    (n4, l4, r4, *_), (n5, l5, r5, *_) = summary
    ax.annotate(f"{n4}, layer {l4}: {r4['normalized_entropy']:.3f}\nbusiest expert {100 * r4['max_load_fraction']:.1f}%",
                xy=(l4, r4["normalized_entropy"]), xytext=(0.6, 0.545), fontsize=11, fontweight="bold", color=INK,
                va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=0, shrinkB=5))
    ax.annotate(f"{n5}, layer {l5}: {r5['normalized_entropy']:.3f}\nbusiest expert {100 * r5['max_load_fraction']:.1f}%",
                xy=(l5, r5["normalized_entropy"]), xytext=(12.4, 0.545), fontsize=11, fontweight="bold", color=INK,
                va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=0, shrinkB=5))
    for a in (ax, right):
        a.axhline(ALARM_ENTROPY, color=ALARM, lw=1.5, zorder=1)
        a.set_ylim(0.5, 1.0)
    right.text(19.8, ALARM_ENTROPY - 0.008, "alarm 0.7", fontsize=11, fontweight="bold", color=ALARM, ha="right",
               va="top")
    ax.set_xlim(-0.6, 25.6)
    ax.set_xticks(range(26))
    ax.tick_params(axis="x", labelsize=9.5)
    label_axes(ax, "Gemma layer", "normalized load entropy")
    right.set_xlim(0, 20)
    label_axes(right, "tokens trained (millions)", "lowest entropy over all layers")

    legend(fig, [line_handle(S4, "Stage 4 (20 MoE layers, MobileMoE-S, pretrained)", marker="D"),
                 line_handle(S5, "Stage 5 (20 MoE layers, MobileMoE-S, random)")], 0.72)
    w, _ = fig.get_size_inches()
    footnote(fig, f"Dead experts: {n4} {summary[0][3]} of {summary[0][4]:,}, {n5} {summary[1][3]} of "
                  f"{summary[1][4]:,}; mean entropy {summary[0][5]:.3f} and {summary[1][5]:.3f}",
             y_in=0.72, x_in=w - 0.4, ha="right")
    save(fig, "routing_stage4_5.png")


def main() -> None:
    pre, rnd = load("stage4_spread_15m"), load("stage5_random_15m")
    s2, s3 = load("stage2_5m"), load("stage3_20m")
    matched_tokens(pre, rnd, s2, s3)
    final_eval(pre, rnd)
    qa(pre, rnd, s3)
    routing(pre, rnd)


if __name__ == "__main__":
    main()
