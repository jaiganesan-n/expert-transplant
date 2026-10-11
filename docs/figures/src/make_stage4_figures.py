"""Stage 4 infographics (PNG, 4,000 px wide) into docs/figures/stage4/.

    python docs/figures/src/make_stage4_figures.py      # the equation card needs pdflatex and pdftoppm

stage4_architecture.png   layout `spread`: Gemma 3 1B keeps the FFNs of layers 2, 6, 10, 15, 19, 23;
                          the other 20 are replaced by MobileMoE-S MoE blocks at the same relative depth
stage4_equations.png      the whole model and its training in two lines

Same visual language as fig1-fig4: green = trainable (W_in, W_out); orange = frozen MobileMoE-S (Model B);
blue = Gemma FFN kept; every other colour = frozen Gemma 3 1B (Model A).
MobileMoE-S-Base facts from its model card (20 layers x 768, 60 routed experts with FFN 384, top-4,
1 shared expert with FFN 1536, sigmoid top-k routing with normalization, 1.3B total / 272M active).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import C, FAINT, INK, MUTED, RESID, Canvas  # noqa: E402

OUT = HERE.parent / "stage4"

D_A, D_S, FFN_A = 1152, 768, 6912
N_EXPERTS, TOP_K, EXPERT_FFN, SHARED_FFN = 60, 4, 384, 1536
L_A, L_S = 26, 20


def spread_map() -> dict[int, int]:
    """Gemma layer <- MobileMoE-S layer j at the same relative depth: round(j * 25 / 19)."""
    return {round(j * (L_A - 1) / (L_S - 1)): j for j in range(L_S)}


MAP = spread_map()
KEPT = [g for g in range(L_A) if g not in MAP]  # 2, 6, 10, 15, 19, 23


def numbers() -> dict:
    proj = 2 * D_A * D_S
    expert = 3 * D_S * EXPERT_FFN
    shared = 3 * D_S * SHARED_FFN
    router = N_EXPERTS * D_S
    return {"proj": proj, "proj_total": L_S * proj, "block": N_EXPERTS * expert + shared + router + N_EXPERTS + D_S,
            "active": TOP_K * expert + shared + router, "ffn_a": 3 * D_A * FFN_A}


# ----------------------------------------------------------------------------------
def architecture() -> None:
    n = numbers()
    kept = ", ".join(map(str, KEPT))
    cv = Canvas(2000, 935)
    cv.title("Stage 4: Gemma 3 1B keeps 6 FFNs, 20 are replaced by MobileMoE-S MoE blocks at the same relative depth")
    cv.text(40, 112, r"$h_t^{\mathrm{out}} = h_t + P_A(F_l(x_t)),\ \ x_t = N_A(h_t);\qquad "
                     r"F_l = \mathrm{FFN}_A^{(l)}\ \mathrm{for}\ l \in \{" + kept + r"\},\qquad "
                     r"F_l = W_{out}^{(l)}\,\mathrm{MoE}_S^{(j)}(N_S(W_{in}^{(l)}x_t))\ \mathrm{for}\ "
                     r"l = \mathrm{round}(25j/19)$", size=18.5)

    # A: the 26-layer stack
    cv.panel(40, 150, 500, 845, "26 DECODER LAYERS · LAYOUT  spread")
    cv.text(270, 197, "↑ final RMSNorm + LM head (Gemma)", size=11, ha="center", color=MUTED)
    top, row = 214, 22
    for g in range(L_A - 1, -1, -1):
        y0 = top + (L_A - 1 - g) * row
        y1 = y0 + row - 3
        cv.text(62, (y0 + y1) / 2, f"L{g}", size=10, color=INK if g in KEPT else MUTED,
                weight="bold" if g in KEPT else "normal")
        if g in KEPT:
            cv.box(110, y0, 470, y1, "ffn", r"$\mathrm{FFN}_A$ · Gemma's own, kept, frozen", size=10, lw=1.3)
        else:
            cv.box(110, y0, 172, y1, "train", r"$W_{in}$", size=10)
            cv.box(176, y0, 404, y1, "moe", f"MoE-S layer {MAP[g]}", size=10, lw=1.3)
            cv.box(408, y0, 470, y1, "train", r"$W_{out}$", size=10)
    cv.text(270, top + L_A * row + 8, "↑ embeddings (Gemma)", size=11, ha="center", color=MUTED)
    cv.text(270, top + L_A * row + 34, "every layer keeps Gemma's attention and norms (not drawn)", size=10.5,
            ha="center", color=FAINT)

    # B: inside a layer
    cv.panel(530, 150, 1450, 845, "INSIDE A LAYER")
    cv.text(560, 202, r"transplanted layer (20 layers), e.g. Gemma layer 3 $\leftarrow$ MobileMoE-S layer 2", size=13.5)
    y = 290
    cv.text(560, y, r"$h_t$", size=16, ha="center")
    cv.arrow([(572, y), (585, y)])
    cv.box(585, y - 30, 655, y + 30, "norm", r"$N_A$", "RMSNorm", size=15, size2=10)
    cv.arrow([(655, y), (680, y)])
    cv.box(680, y - 34, 790, y + 34, "train", r"$W_{in}$", rf"${D_A}\to{D_S}$", size=17, size2=10.5)
    cv.arrow([(790, y), (815, y)])
    cv.box(815, y - 34, 1085, y + 34, "moe", r"$N_S$ + $\mathrm{MoE}_S$", "top-4 of 60 experts + 1 shared", size=16,
           size2=10.5)
    cv.arrow([(1085, y), (1110, y)])
    cv.box(1110, y - 34, 1220, y + 34, "train", r"$W_{out}$", rf"${D_S}\to{D_A}$", size=17, size2=10.5)
    cv.arrow([(1220, y), (1245, y)])
    cv.box(1245, y - 30, 1315, y + 30, "norm", r"$P_A$", "RMSNorm", size=15, size2=10)
    cv.arrow([(1315, y), (1335, y)])
    cv.plus(1350, y)
    cv.arrow([(1365, y), (1400, y)])
    cv.text(1422, y, r"$h^{\mathrm{out}}_t$", size=15, ha="center")
    cv.arrow([(568, y - 12), (568, 236), (1350, 236), (1350, y - 15)], color=RESID, lw=1.8)
    cv.text(960, 226, "residual", size=11.5, ha="center", color=RESID)
    ec = C["moe"][1]
    cv.ax.add_patch(FancyBboxPatch((680, 360), 540, 196, boxstyle="round,pad=0,rounding_size=8", fc="#FFF6EC",
                                   ec=ec, lw=1.4, ls=(0, (5, 3)), zorder=1))
    cv.arrow([(950, y + 34), (950, 360)], color=ec, dashed=True)
    cv.text(700, 382, "MobileMoE-S MoE block (frozen), on $u = N_S(W_{in}x_t)$", size=12.5, weight="bold",
            color="#9A5A12")
    lines = [
        r"router: $s = \sigma(W_r\,u) \in (0,1)^{60}$, pick the top 4 of $s + b$",
        r"weights $g_e$: the 4 scores normalized to sum 1, times $\lambda$",
        r"60 routed experts, each a gated FFN $768\to384\to768$",
        r"1 shared expert, always on, gated FFN $768\to1536\to768$",
        f"{n['block'] / 1e6:.1f}M parameters per block, {n['active'] / 1e6:.1f}M active per token",
    ]
    for k, s in enumerate(lines):
        cv.text(712, 414 + 28 * k, "·  " + s, size=12, color="#333333")
    cv.text(950, 582, rf"trainable: $2\times{D_A}\times{D_S}$ = {n['proj']:,} per layer, "
                      rf"$\times 20$ = {n['proj_total'] / 1e6:.1f}M", size=12.5, ha="center", color="#1E7A3C")
    cv.text(560, 628, f"kept layer (6 layers: {kept}): Gemma's own FFN, untouched", size=13.5)
    y = 712
    cv.text(560, y, r"$h_t$", size=16, ha="center")
    cv.arrow([(572, y), (585, y)])
    cv.box(585, y - 30, 655, y + 30, "norm", r"$N_A$", "RMSNorm", size=15, size2=10)
    cv.arrow([(655, y), (680, y)])
    cv.box(680, y - 34, 1220, y + 34, "ffn", rf"$\mathrm{{FFN}}_A^{{(l)}}$:  GeGLU  ${D_A}\to{FFN_A}\to{D_A}$",
           f"frozen, no projections ({n['ffn_a'] / 1e6:.1f}M parameters, as in Gemma)", size=15, size2=10.5)
    cv.arrow([(1220, y), (1245, y)])
    cv.box(1245, y - 30, 1315, y + 30, "norm", r"$P_A$", "RMSNorm", size=15, size2=10)
    cv.arrow([(1315, y), (1335, y)])
    cv.plus(1350, y)
    cv.arrow([(1365, y), (1400, y)])
    cv.text(1422, y, r"$h^{\mathrm{out}}_t$", size=15, ha="center")
    cv.arrow([(568, y - 12), (568, 662), (1350, 662), (1350, y - 15)], color=RESID, lw=1.8)
    cv.text(990, 805, "Training signal for every layer: CE + 1.0 · KL to unmodified Gemma (all 26 FFNs on).",
            size=12, ha="center", color="#333333")
    cv.text(990, 827, "Only the green projections receive gradients; the 6 kept FFNs stay frozen.", size=12,
            ha="center", color="#333333")

    # C: the relative-depth map and key numbers
    cv.panel(1480, 150, 1960, 845, "LAYER MAP · RELATIVE DEPTH")
    cv.text(1500, 200, r"MobileMoE-S layer $j$ $\rightarrow$ Gemma layer $\mathrm{round}(25j/19)$", size=12.5)
    x0, cw = 1512, 16
    yg, ys = 262, 372  # Gemma row, MobileMoE-S row
    cv.text(x0, yg - 18, "Gemma 3 1B, 26 layers", size=11, color=MUTED, weight="bold")
    cv.text(x0, ys + 50, "MobileMoE-S, 20 layers", size=11, color=MUTED, weight="bold")
    for g in range(L_A):
        kind = "ffn" if g in KEPT else "moe"
        fc, ecc = C[kind]
        cv.ax.add_patch(FancyBboxPatch((x0 + g * cw, yg), cw - 2, 24, boxstyle="round,pad=0,rounding_size=2",
                                       fc=fc, ec=ecc, lw=1.1, zorder=2))
        cv.text(x0 + g * cw + (cw - 2) / 2, yg + 12, str(g), size=6.3, ha="center",
                color="#2F4F7F" if g in KEPT else "#7A4A10")
    for j in range(L_S):
        xs = x0 + j * (L_A - 1) / (L_S - 1) * cw  # same relative depth on the same axis
        cv.ax.add_patch(FancyBboxPatch((xs, ys), cw - 2, 24, boxstyle="round,pad=0,rounding_size=2",
                                       fc=C["moe"][0], ec=C["moe"][1], lw=1.1, zorder=2))
        cv.text(xs + (cw - 2) / 2, ys + 12, str(j), size=6.3, ha="center", color="#7A4A10")
        g = round(j * (L_A - 1) / (L_S - 1))
        cv.ax.plot([xs + (cw - 2) / 2, x0 + g * cw + (cw - 2) / 2], [ys, yg + 24], color="#E8892B", lw=0.9,
                   alpha=0.8, zorder=1)
    cv.text(x0, 455, f"kept Gemma FFNs (blue): layers {kept}", size=11.5, color="#2F4F7F")
    cv.text(x0, 478, "about every 4th layer, spread over the depth", size=11, color=FAINT)
    info = [
        ("MobileMoE-S-Base (Model B)", True),
        ("20 layers × 768; 60 routed experts (FFN 384),", False),
        ("top-4, + 1 shared expert (FFN 1536)", False),
        ("1.3B total, 272M active; bf16", False),
        ("Trainable", True),
        (f"{n['proj_total'] / 1e6:.1f}M = 20 × (1152×768 + 768×1152)", False),
        ("Stages 1–3: 61.3M (26 layers, 1152 ↔ 1024)", False),
        ("Training", True),
        ("ridge init on the 20 mapped layers, then", False),
        ("end to end, Stage 3's data and settings", False),
    ]
    yy = 530
    for s, bold in info:
        if bold and yy > 540:
            yy += 10
        cv.text(x0, yy, s, size=11.5 if not bold else 12, weight="bold" if bold else "normal",
                color=INK if bold else "#333333")
        yy += 23

    cv.legend(870, [("train", r"trainable: $W_{in}$, $W_{out}$", False),
                    ("moe", "frozen, from MobileMoE-S (Model B)", False),
                    ("ffn", "Gemma FFN kept, frozen", False),
                    ("norm", "other colours: frozen Gemma 3 1B (Model A)", False)])
    cv.text(40, 906, "Data: the MobileMoE pre-training mix of Stages 1–3, read in the same order as Stages 2–3, each token "
                     "once. Ridge init recomputed for MobileMoE-S; no local matching.", size=11.5, color=FAINT)
    OUT.mkdir(parents=True, exist_ok=True)
    cv.save("stage4/stage4_architecture.png")


# ----------------------------------------------------------------------------------
CARD = r"""\documentclass[border=40pt]{standalone}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{xcolor}
\usepackage[scaled=0.95]{helvet}
\definecolor{bg}{HTML}{F8F8F8}
\definecolor{muted}{HTML}{6B7280}
\pagecolor{bg}
\begin{document}
\begin{minipage}{30cm}
{\sffamily\bfseries\Large\color{muted} Stage 4: Gemma 3 1B with 20 MobileMoE-S MoE layers (spread layout)}\par\vspace{-14pt}
{\LARGE
\begin{gather*}
h^{\mathrm{out}}_t = h_t + P_A\big(F_l(x_t)\big), \qquad
F_l = \begin{cases}
\mathrm{FFN}_A^{(l)}, & l \in \{2, 6, 10, 15, 19, 23\}\\[2pt]
W_{out}^{(l)}\,\mathrm{MoE}_S^{(j)}\big(N_S(W_{in}^{(l)}\,\cdot\,)\big), & l = \mathrm{round}(25j/19),\ j = 0,\dots,19
\end{cases}
\\[12pt]
\mathrm{MoE}_S(u) = S(u) + \sum_{e \in \mathrm{Top4}(\sigma(W_r u) + b)} g_e\,E_e(u), \qquad
\mathcal{L} = \mathrm{CE}(y, p_{\mathrm{hyb}}) + \mathrm{KL}(p_A \,\Vert\, p_{\mathrm{hyb}}), \qquad
\underbrace{20 \cdot 2 \cdot 1152 \cdot 768}_{\text{trainable}} = 35.4\mathrm{M}
\end{gather*}}
\end{minipage}
\end{document}
"""


def equations() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "equations" / "stage4_equations.tex").write_text(CARD)
    work = Path(tempfile.mkdtemp())
    try:
        (work / "card.tex").write_text(CARD)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "card.tex"], cwd=work,
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "4000", "-scale-to-y", "-1", "card.pdf",
                        str(OUT / "stage4_equations")], cwd=work, check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", OUT / "stage4_equations.png")


if __name__ == "__main__":
    assert KEPT == [2, 6, 10, 15, 19, 23], KEPT
    architecture()
    equations()
