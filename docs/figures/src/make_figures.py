"""Draw the four architecture figures (PNG, 4,000 px wide) into docs/figures/.

    python docs/figures/src/make_figures.py

fig1_end_to_end.png        the hybrid from token ids to next-token probabilities
fig2_replace_layer.png     one transplanted layer, inside Gemma's sandwich norms
fig3_moe_routing.png       the transplanted MoE block routing one token
fig4_training_pipeline.png ridge init, local matching, end to end

Colour code: green = trainable (W_in, W_out); orange = frozen MobileMoE-M (Model B);
every other colour = frozen Gemma 3 1B (Model A); grey dashed = removed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Sans"

BG = "#F8F8F8"
PANEL = ("#F2F3F5", "#D5D8DC")
INK = "#222222"
MUTED = "#6B7280"
FAINT = "#8A8F98"
RESID = "#1F4E79"
C = {
    "train": ("#D5F0DD", "#2E9E4F"),
    "moe": ("#FDE3C8", "#E8892B"),
    "norm": ("#E8E3F5", "#A08ED6"),
    "attn": ("#FFF3B0", "#D9B92B"),
    "embed": ("#F5D7DF", "#D97A8E"),
    "ffn": ("#DCE6F2", "#8FAAD6"),
    "gemma": ("#DCE6F2", "#8FAAD6"),
    "plain": ("#FFFFFF", "#C8CCD2"),
    "removed": ("#F2F2F2", "#BDBDBD"),
    "loss": ("#FBDDE0", "#D9534F"),
    "token": ("#DCE6F2", "#8FAAD6"),
    "hot": ("#FDE0E0", "#D9534F"),
}


class Canvas:
    """Pixel-like coordinates: x 0..W, y 0..H downward, at 2 px per unit in the PNG."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=200, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=13, color=INK, ha="left", va="center", weight="normal", **kw):
        kw.setdefault("zorder", 5)
        self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **kw)

    def title(self, s):
        self.text(40, 38, s, size=21, weight="bold")
        self.ax.plot([40, self.w - 40], [75, 75], color="#D0D3D8", lw=1.2)

    def panel(self, x0, y0, x1, y1, label=None):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=10",
                                         fc=PANEL[0], ec=PANEL[1], lw=1.4, zorder=0.5))
        if label:
            self.text(x0 + 20, y0 + 22, label, size=11.5, color=MUTED, weight="bold")

    def box(self, x0, y0, x1, y1, kind, line1, line2=None, size=14, size2=10.5, dashed=False, lw=1.6, color=INK):
        fc, ec = C[kind]
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=6",
                                         fc=fc, ec=ec, lw=lw if kind != "train" else 2.6,
                                         ls=(0, (5, 3)) if dashed else "-", zorder=2))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if line2:
            self.text(cx, cy - 11, line1, size=size, ha="center", color=color)
            self.text(cx, cy + 13, line2, size=size2, ha="center", color=MUTED if color == INK else color)
        else:
            self.text(cx, cy, line1, size=size, ha="center", color=color)

    def arrow(self, pts, color="#3A3F47", lw=1.6, dashed=False, head=True):
        xs, ys = zip(*pts)
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-", solid_capstyle="butt", zorder=3)
        a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=7,head_width=4" if head else "-",
                            color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0,
                            mutation_scale=1.2, zorder=3)
        self.ax.add_patch(a)

    def plus(self, x, y, r=15, ec="#A08ED6", fc="#ECE7F7"):
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.6, zorder=4))
        self.text(x, y + 1, "+", size=16, ha="center")

    def legend(self, y, items, x=40):
        for kind, label, dashed in items:
            fc, ec = C[kind]
            self.ax.add_patch(FancyBboxPatch((x, y - 9), 26, 18, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec,
                                             lw=2 if kind == "train" else 1.5, ls=(0, (3, 2)) if dashed else "-"))
            self.text(x + 36, y, label, size=12, color="#333333")
            x += 36 + 8.3 * len(label.replace("$", "").replace("_", "").replace("\\", "")) + 40

    def save(self, name):
        path = OUT / name
        self.fig.savefig(path, dpi=200, facecolor=BG)
        plt.close(self.fig)
        print("wrote", path)


# ----------------------------------------------------------------------------------
def fig1():
    cv = Canvas(2000, 820)
    cv.title("Expert Transplant: Gemma 3 1B with MobileMoE experts, end to end")
    cv.text(40, 118, r"$X^{(0)} = \sqrt{1152}\;E_A[t]\;,\qquad X^{(l)} = \mathrm{Layer}^{\mathrm{hyb}}_{l}\,(\,X^{(l-1)}\,)\;,"
                     r"\qquad P = \mathrm{softmax}(\,N_f(X^{(26)})\,E_A^{\top}\,)$", size=21)
    # tokens
    toks = ["<bos>", "The", "capital", "of", "France", "is"]
    cv.text(124, 244, r"$t$", size=18, ha="center")
    for k, t in enumerate(toks):
        kind = "hot" if k == 5 else "token"
        cv.box(52, 268 + 44 * k, 196, 312 + 44 * k, kind, t, size=12.5)
    cv.text(124, 558, "6 token ids", size=14, ha="center", color="#333333")
    cv.text(124, 584, "Gemma tokenizer", size=11.5, ha="center", color=FAINT)
    cv.arrow([(205, 400), (252, 400)])
    # embeddings
    cv.panel(262, 172, 572, 700, "EMBEDDINGS · GEMMA")
    cv.box(284, 236, 550, 318, "embed", r"$E_A[t] \in \mathbb{R}^{6\times1152}$", r"word, from $262{,}144\times1152$", size=15, size2=12)
    cv.text(417, 344, r"$\times$", size=18, ha="center", color=MUTED)
    cv.box(284, 368, 550, 450, "embed", r"$\sqrt{1152}\approx 33.94$", "Gemma's embedding scale", size=15, size2=12)
    cv.box(284, 500, 550, 576, "removed", "no position table", "RoPE, applied inside attention", size=13, size2=11, dashed=True)
    cv.arrow([(550, 409), (628, 409), (628, 290), (698, 290)])
    cv.text(600, 394, r"$6\times1152$", size=11, ha="center", color=MUTED)
    # hybrid decoder layer
    cv.panel(652, 172, 1652, 700, "HYBRID DECODER LAYER × 26")
    cv.box(700, 258, 810, 322, "norm", "RMSNorm", size=13)
    cv.box(845, 254, 1145, 326, "attn", "Multi-Query Attention", r"4 heads, 1 KV head, $d_k = 256$", size=14, size2=11)
    cv.box(1180, 258, 1290, 322, "norm", "RMSNorm", size=13)
    cv.plus(1330, 290)
    cv.arrow([(810, 290), (845, 290)])
    cv.arrow([(1145, 290), (1180, 290)])
    cv.arrow([(1290, 290), (1315, 290)])
    cv.arrow([(675, 290), (675, 226), (1330, 226), (1330, 275)], color=RESID, lw=1.8)
    cv.text(1000, 213, "residual", size=12, ha="center", color=RESID)
    cv.arrow([(1345, 290), (1384, 290), (1384, 362), (675, 362), (675, 512), (698, 512)])
    cv.text(1030, 349, r"$h: 6\times1152$, after attention", size=12, ha="center", color="#333333")
    cv.box(700, 480, 800, 544, "norm", r"$N_A$", "RMSNorm", size=15, size2=10.5)
    # removed FFN (dashed, grey)
    cv.box(850, 410, 1420, 470, "removed", r"Gemma $\mathrm{FFN}_A$, removed:  GeGLU  $1152\to6912\to1152$",
           "the transplant takes its slot in every transplanted layer", size=13, size2=10.5, dashed=True, color="#8A8A8A")
    cv.arrow([(825, 512), (825, 440), (848, 440)], color="#B0B0B0", dashed=True)
    # transplant row
    cv.arrow([(800, 512), (825, 512), (825, 578), (850, 578)])
    cv.box(850, 544, 980, 612, "train", r"$W_{in}$", r"$1152\to1024$", size=17, size2=10.5)
    cv.arrow([(980, 578), (1010, 578)])
    cv.box(1010, 544, 1260, 612, "moe", r"$N_B$ + $\mathrm{MoE}_B$", "top-4 of 60 experts + 1 shared", size=16, size2=10.5)
    cv.arrow([(1260, 578), (1290, 578)])
    cv.box(1290, 544, 1420, 612, "train", r"$W_{out}$", r"$1024\to1152$", size=17, size2=10.5)
    cv.arrow([(1420, 578), (1452, 578), (1452, 512), (1488, 512)])
    cv.box(1490, 480, 1580, 544, "norm", r"$P_A$", "RMSNorm", size=15, size2=10.5)
    cv.arrow([(1580, 512), (1600, 512)])
    cv.plus(1615, 512)
    cv.arrow([(675, 512), (675, 648), (1615, 648), (1615, 527)], color=RESID, lw=1.8)
    cv.text(760, 635, "residual", size=12, ha="center", color=RESID)
    cv.arrow([(1630, 512), (1725, 512)])
    cv.text(1690, 497, r"$6\times1152$", size=12, ha="center", color=MUTED)
    cv.text(1152, 676, r"each of the 26 layers has its own $W_{in}$, $W_{out}$   ·   2.36M trainable parameters per layer",
            size=13, ha="center", color="#333333")
    # LM head
    cv.panel(1732, 172, 1960, 700, "LM HEAD · GEMMA")
    cv.box(1752, 236, 1940, 302, "norm", "final RMSNorm", r"$N_f$", size=14, size2=12)
    cv.arrow([(1846, 302), (1846, 342)])
    cv.box(1752, 344, 1940, 420, "embed", r"$\times\,E_A^{\top}$ (tied)", r"$1152\to262{,}144$", size=15, size2=11)
    cv.arrow([(1846, 420), (1846, 460)])
    cv.box(1752, 462, 1940, 528, "norm", "softmax", "per row", size=14, size2=12)
    cv.text(1846, 568, r"$P \in \mathbb{R}^{6\times262{,}144}$", size=15, ha="center")
    cv.text(1846, 604, "row 6 → “ Paris”", size=13, ha="center", color="#333333")
    cv.legend(732, [("train", r"trainable: $W_{in}$, $W_{out}$", False),
                    ("moe", "frozen, from MobileMoE-M (Model B)", False),
                    ("norm", "every other colour: frozen, from Gemma 3 1B (Model A)", False),
                    ("removed", "removed", True)])
    cv.text(40, 770, "Running example: n = 6 tokens, d_A = 1152, d_B = 1024, 60 routed experts (top-4) + 1 shared expert, "
                     "|V| = 262,144. Every arrow carries one row per token.", size=12, color=FAINT)
    cv.text(40, 796, "Only the green projections train: 2.36M parameters per layer, 61.3M across all 26 layers. "
                     "Token splits are illustrative.", size=12, color=FAINT)
    cv.save("fig1_end_to_end.png")


# ----------------------------------------------------------------------------------
def fig2():
    cv = Canvas(2000, 800)
    cv.title("One transplanted layer: Gemma's FFN replaced by MobileMoE's MoE block")
    cv.panel(40, 100, 1380, 700, "REPLACE  ·  TESTS H1, STITCHABILITY")
    cv.text(710, 168, r"$h_t^{\mathrm{out}} = h_t + P_A(\,W_{out}\;\mathrm{MoE}_B(N_B(W_{in}\,x_t))\,),\qquad x_t = N_A(h_t)$",
            size=20, ha="center")
    # sandwich bracket
    cv.text(710, 222, r"Gemma's sandwich: $N_A$ before the FFN slot, $P_A$ after it", size=12.5, ha="center", color=FAINT)
    cv.ax.plot([150, 150, 1270, 1270], [250, 238, 238, 250], color="#7A7F87", lw=1.4)
    # input
    cv.text(95, 378, r"$h_t$", size=17, ha="center")
    cv.box(130, 368, 230, 432, "norm", r"$N_A$", "RMSNorm", size=17, size2=10.5)
    cv.arrow([(70, 580), (70, 400), (128, 400)], color=RESID, lw=1.8)
    # removed FFN
    cv.box(330, 286, 900, 346, "removed", r"Gemma $\mathrm{FFN}_A$, removed", r"GeGLU  $1152\to6912\to1152$ · 23.9M parameters",
           size=15, size2=11, dashed=True, color="#8A8A8A")
    cv.arrow([(255, 400), (255, 316), (328, 316)], color="#B0B0B0", dashed=True)
    # transplant row
    cv.arrow([(230, 400), (255, 400), (255, 488), (300, 488)])
    cv.text(262, 418, r"$x_t \in \mathbb{R}^{1152}$", size=12.5, color="#333333")
    cv.box(300, 454, 440, 522, "train", r"$W_{in}$", r"$1024\times1152$", size=18, size2=11)
    cv.arrow([(440, 488), (480, 488)])
    cv.text(460, 440, r"$\mathbb{R}^{1024}$", size=12, ha="center", color=MUTED)
    cv.box(480, 454, 580, 522, "moe", r"$N_B$", "RMSNorm", size=17, size2=10.5)
    cv.arrow([(580, 488), (620, 488)])
    cv.box(620, 454, 790, 522, "moe", r"$\mathrm{MoE}_B$", "4 of 60 + shared", size=17, size2=10.5)
    cv.arrow([(790, 488), (830, 488)])
    cv.text(810, 440, r"$m_t \in \mathbb{R}^{1024}$", size=12, ha="center", color=MUTED)
    cv.box(830, 454, 970, 522, "train", r"$W_{out}$", r"$1152\times1024$", size=18, size2=11)
    cv.text(380, 545, "ridge init", size=12.5, ha="center", color="#2E9E4F", weight="bold")
    cv.text(900, 545, "ridge init", size=12.5, ha="center", color="#2E9E4F", weight="bold")
    cv.arrow([(970, 488), (1050, 488), (1050, 400), (1098, 400)])
    cv.text(1012, 470, r"$\mathbb{R}^{1152}$", size=12, ha="center", color=MUTED)
    cv.box(1100, 368, 1200, 432, "norm", r"$P_A$", "RMSNorm", size=17, size2=10.5)
    cv.arrow([(1200, 400), (1235, 400)])
    cv.plus(1250, 400)
    cv.arrow([(1250, 385), (1250, 330)])
    cv.text(1250, 312, r"$h_t^{\mathrm{out}}$", size=17, ha="center")
    cv.arrow([(70, 580), (1250, 580), (1250, 415)], color=RESID, lw=1.8)
    cv.text(660, 567, "residual", size=12.5, ha="center", color=RESID)
    cv.text(70, 625, r"Init: $W_{in}$ by ridge to MobileMoE's own MoE inputs (RMS-normalized); $W_{out}$ by ridge to $\mathrm{FFN}_A$ outputs.",
            size=13, color="#333333")
    cv.text(70, 655, r"Train: local matching (match $\mathrm{FFN}_A$ at every layer, one Gemma pass), then end to end with $\beta = 1$.",
            size=13, color="#333333")
    # side column: what each part is
    cv.panel(1410, 100, 1960, 700, "PER TRANSPLANTED LAYER")
    rows = [
        ("train", "Trainable", r"$W_{in}$: $1024\times1152$,  $W_{out}$: $1152\times1024$", "2.36M parameters"),
        ("moe", "Frozen, from MobileMoE-M", "$N_B$, router $60\\times1024$, 60 experts ($1024\\to512\\to1024$),\nshared expert ($1024\\to2048\\to1024$)", "100.7M parameters, 12.6M active per token"),
        ("removed", "Removed from Gemma's forward pass", "$\\mathrm{FFN}_A$ stays loaded, frozen: the local-matching target\nand the KL teacher only", "23.9M parameters"),
        ("norm", "Kept, frozen, from Gemma 3 1B", "$N_A$, $P_A$, attention, embeddings, LM head", "unchanged"),
    ]
    y = 150
    for kind, head, body, foot in rows:
        fc, ec = C[kind]
        cv.ax.add_patch(FancyBboxPatch((1432, y), 506, 118, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec,
                                       lw=2.4 if kind == "train" else 1.4, ls=(0, (5, 3)) if kind == "removed" else "-"))
        cv.text(1450, y + 22, head, size=13, weight="bold", color="#333333")
        cv.text(1450, y + 58, body, size=11.5, color="#333333", linespacing=1.4)
        cv.text(1450, y + 100, foot, size=11.5, color=MUTED)
        y += 134
    cv.legend(732, [("train", r"trainable: $W_{in}$, $W_{out}$", False), ("moe", "frozen, from MobileMoE-M", False),
                    ("norm", "frozen, from Gemma 3 1B", False), ("removed", "removed from the forward pass", True)])
    cv.text(40, 774, "Shapes are per token. With n tokens every arrow carries an n-row matrix; ⊕ adds row by row.",
            size=12, color=FAINT)
    cv.save("fig2_replace_layer.png")


# ----------------------------------------------------------------------------------
def fig3():
    cv = Canvas(2000, 800)
    cv.title("Inside the transplanted MoE block: routing one token")
    cv.text(40, 116, r"$\hat u_t = N_B(W_{in}\,x_t)\;,\qquad s_t = \sigma(W_r\,\hat u_t)\;,\qquad \mathcal{T}_t = \mathrm{TopK}(s_t + b,\,4)\;,"
                     r"\qquad m_t = S(\hat u_t) + \sum_{e\,\in\,\mathcal{T}_t} g_{t,e}\,E_e(\hat u_t)$", size=20)
    # one token
    cv.panel(40, 180, 310, 690, "ONE TOKEN")
    cv.box(65, 222, 285, 286, "token", "token 5: “ France”", r"$x_t = N_A(h_t) \in \mathbb{R}^{1152}$", size=13, size2=10.5)
    cv.arrow([(175, 286), (175, 314)])
    cv.box(105, 316, 245, 372, "train", r"$W_{in}$", r"$1152\to1024$", size=16, size2=10)
    cv.arrow([(175, 372), (175, 400)])
    cv.box(105, 402, 245, 458, "moe", r"$N_B$", "RMSNorm", size=16, size2=10)
    cv.arrow([(175, 458), (175, 488)])
    cv.box(65, 490, 285, 550, "plain", r"$\hat u_t \in \mathbb{R}^{1024}$", size=15)
    cv.arrow([(285, 520), (344, 520)])
    cv.box(346, 468, 516, 572, "moe", "", size=16)
    cv.text(431, 496, r"Router $W_r$", size=16, ha="center")
    cv.text(431, 526, r"$60\times1024$", size=11, ha="center", color=MUTED)
    cv.text(431, 548, "sigmoid per expert", size=11, ha="center", color=MUTED)
    cv.arrow([(516, 520), (564, 520)])
    # scores
    cv.panel(566, 180, 1160, 690, "ROUTER SCORES FOR ALL 60 EXPERTS")
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.05, 0.63, 60)
    top = {7: 0.91, 23: 0.86, 41: 0.79, 52: 0.74}
    for e, v in top.items():
        scores[e] = v
    x0, x1, yb, yt = 620, 1138, 626, 296
    cv.ax.plot([x0, x0], [yb, yt], color="#555555", lw=1.2)
    cv.ax.plot([x0, x1], [yb, yb], color="#555555", lw=1.2)
    for val, lab in ((0, "0"), (0.5, "0.5"), (1, "1")):
        y = yb - (yb - yt) * val
        cv.ax.plot([x0 - 5, x0], [y, y], color="#555555", lw=1.0)
        cv.text(x0 - 10, y, lab, size=11, ha="right", color="#555555")
    bw = (x1 - x0) / 60
    for e, v in enumerate(scores):
        hot = e in top
        h = (yb - yt) * v
        cv.ax.add_patch(plt.Rectangle((x0 + e * bw + 1.2, yb - h), bw - 2.4, h, fc="#FDE3C8" if hot else "#E3E5E8",
                                      ec="#E8892B" if hot else "#C3C7CD", lw=1.2 if hot else 0.6))
        if hot:
            cv.text(x0 + e * bw + bw / 2, yb - h - 16, str(e), size=12, ha="center", color="#E8892B", weight="bold")
    cut = yb - (yb - yt) * 0.735
    cv.ax.plot([x0, x1], [cut, cut], color="#E8892B", lw=1.3, ls=(0, (4, 3)))
    cv.text(760, cut - 14, "top-4 cut", size=11.5, ha="center", color="#E8892B")
    cv.text(622, 645, "expert 0", size=11.5, color="#333333")
    cv.text(1136, 645, "59", size=11.5, ha="right", color="#333333")
    cv.text(880, 670, r"each $s_{t,e} \in (0,1)$ on its own; the 60 scores need not sum to 1", size=12, ha="center", color="#333333")
    cv.arrow([(1160, 520), (1194, 520)])
    # top-4 table
    cv.panel(1196, 180, 1446, 690, "TOP-4 WEIGHTS")
    cv.text(1248, 250, r"$e$", size=14, ha="center", color=MUTED)
    cv.text(1320, 250, r"$s_{t,e}$", size=14, ha="center", color=MUTED)
    cv.text(1398, 250, r"$g_{t,e}$", size=14, ha="center", color=MUTED)
    cv.ax.plot([1216, 1426], [272, 272], color="#C8CCD2", lw=1)
    total = sum(top.values())
    for k, (e, v) in enumerate(top.items()):
        y = 304 + 44 * k
        cv.text(1248, y, str(e), size=15, ha="center", color="#E8892B", weight="bold")
        cv.text(1320, y, f"{v:.2f}", size=15, ha="center")
        cv.text(1398, y, f"{2.5 * v / total:.2f}", size=15, ha="center", weight="bold")
    cv.ax.plot([1216, 1426], [466, 466], color="#C8CCD2", lw=1)
    cv.text(1320, 488, f"{total:.2f}", size=14, ha="center", color="#333333")
    cv.text(1398, 488, "2.50", size=14, ha="center", color="#333333")
    cv.text(1321, 565, r"$g_{t,e} = 2.5\,\frac{s_{t,e}}{\sum_{e'\in\mathcal{T}_t} s_{t,e'}}$", size=17, ha="center")
    cv.text(1321, 634, "56 experts stay idle", size=12.5, ha="center", color="#333333")
    cv.text(1321, 658, "for this token", size=12.5, ha="center", color="#333333")
    # experts
    cv.panel(1482, 180, 1792, 690, "EXPERTS THAT RUN")
    cv.text(1463, 504, r"$\hat u_t$", size=13, ha="center", color=MUTED)
    cv.arrow([(1446, 520), (1504, 520)], head=False)
    ys = [256, 333, 410, 487]
    for (e, v), y in zip(top.items(), ys):
        cv.arrow([(1504, 520), (1504, y), (1522, y)])
        cv.box(1524, y - 30, 1720, y + 30, "moe", rf"$E_{{{e}}}$", r"$1024\to512\to1024$", size=15, size2=10.5)
        cv.text(1724, y - 16, f"× {2.5 * v / total:.2f}", size=11.5, weight="bold")
        cv.arrow([(1720, y), (1745, y), (1825, 515)], lw=1.3)
    cv.text(1622, 535, "always on", size=12, ha="center", color="#E8892B", weight="bold")
    cv.arrow([(1504, 520), (1504, 581), (1522, 581)])
    cv.box(1524, 551, 1720, 611, "moe", r"$S$, shared", r"$1024\to2048\to1024$", size=15, size2=10.5)
    cv.text(1726, 596, "× 1", size=11.5, weight="bold")
    cv.arrow([(1720, 581), (1745, 581), (1825, 525)], lw=1.3)
    cv.plus(1840, 519, ec="#E8892B", fc="#FDE3C8")
    cv.arrow([(1855, 519), (1878, 519)])
    cv.text(1885, 506, r"$m_t \in \mathbb{R}^{1024}$", size=13)
    cv.text(1915, 534, r"to $W_{out}$", size=12, ha="center", color="#333333")
    cv.legend(728, [("train", r"trainable: $W_{in}$", False), ("moe", "frozen, from MobileMoE-M", False),
                    ("token", "Gemma side", False)])
    cv.text(40, 766, "Scores and expert indices are illustrative. MobileMoE (code read at M0) picks the top 4 of sigmoid scores plus a fixed per-expert bias b, "
                     "then normalizes the chosen four's scores and scales them by 2.5.", size=12, color=FAINT)
    cv.text(40, 790, "Per token the block runs 4 routed experts plus the shared one: about 12.6M of its 100.7M parameters (3-matrix gated experts, confirmed at M0).", size=12, color=FAINT)
    cv.save("fig3_moe_routing.png")


# ----------------------------------------------------------------------------------
def fig4():
    cv = Canvas(2000, 820)
    cv.title("Training pipeline: ridge init, then local matching, then end to end")
    cv.text(40, 112, r"$W_{in}, W_{out} \leftarrow \mathrm{ridge}\;,\qquad \mathcal{L}_1 = \frac{\Vert\hat y - y\Vert^2}{\Vert y\Vert^2}\;,"
                     r"\qquad \mathcal{L}_2 = \mathrm{CE}(y,\,p_{hyb}) + \beta\,\mathrm{KL}(p_A\,\Vert\,p_{hyb})$", size=21)
    # panel 1: ridge
    cv.panel(40, 162, 620, 728, "1 · RIDGE INIT, ONCE PER LAYER SET")
    cv.box(70, 212, 590, 270, "plain", "calibration text", "2M tokens of MobileMoE's pre-training mix, held out", size=14, size2=11)
    cv.arrow([(195, 270), (195, 304)])
    cv.arrow([(465, 270), (465, 304)])
    cv.box(70, 306, 320, 376, "gemma", "Gemma reads it", r"$X$: rows of $N_A(h)$ at layer $i$", size=14, size2=11)
    cv.box(340, 306, 590, 376, "moe", "MobileMoE reads it", r"$Y$: its MoE input at layer $j$", size=14, size2=11)
    cv.arrow([(195, 376), (195, 410)])
    cv.arrow([(465, 376), (465, 410)])
    cv.box(70, 412, 590, 478, "plain", "keep aligned positions", "both tokenizers end a token at the same character", size=14, size2=11)
    cv.arrow([(330, 478), (330, 512)])
    cv.box(70, 514, 590, 580, "plain", "accumulate in fp64, never store activations",
           r"$X^{\top}X$: $1152\times1152$      $X^{\top}Y$: $1152\times1024$", size=14, size2=11)
    cv.arrow([(330, 580), (330, 614)])
    cv.box(70, 616, 590, 688, "train", r"$W_{in} = [\,(X^{\top}X+\lambda I)^{-1}X^{\top}Y\,]^{\top}$",
           r"$\lambda$ picked on a held-out tenth of the calibration batches", size=15, size2=11)
    cv.text(70, 708, r"then $W_{out}$: ridge from $\mathrm{MoE}_B$ outputs to $\mathrm{FFN}_A$ outputs, $W_{in}$ fixed",
            size=11.5, color=MUTED)
    cv.arrow([(620, 450), (708, 450)])
    cv.text(664, 435, "then", size=12.5, ha="center", color="#333333")
    # panel 2: local matching
    cv.panel(710, 162, 1290, 728, "2 · LOCAL MATCHING, EVERY LAYER IN ONE PASS")
    cv.box(740, 212, 1260, 270, "gemma", r"Gemma, own FFNs in place, no grad", r"$x_i = N_A(h_i)$ and $y_i = \mathrm{FFN}_A(x_i)$ read at every layer $i$",
           size=14, size2=11)
    cv.arrow([(840, 270), (840, 304)])
    cv.arrow([(1110, 270), (1110, 304)])
    cv.box(740, 306, 940, 542, "gemma", "", size=13)
    cv.text(840, 380, "target", size=13, ha="center", color="#333333")
    cv.text(840, 420, r"$y_i = \mathrm{FFN}_A(x_i)$", size=15, ha="center")
    cv.text(840, 460, "Gemma's own FFN,", size=11.5, ha="center", color=MUTED)
    cv.text(840, 480, "frozen", size=11.5, ha="center", color=MUTED)
    cv.box(960, 306, 1260, 366, "train", r"$W_{in}$", size=16)
    cv.arrow([(1110, 366), (1110, 394)])
    cv.box(960, 396, 1260, 452, "moe", r"$N_B$ + $\mathrm{MoE}_B$", size=15)
    cv.arrow([(1110, 452), (1110, 480)])
    cv.box(960, 482, 1260, 542, "train", r"$W_{out}$  →  prediction $\hat y_i$", size=15)
    cv.arrow([(840, 542), (840, 586)])
    cv.arrow([(1110, 542), (1110, 586)])
    cv.box(740, 588, 1260, 656, "loss", r"$\mathcal{L}_1 = \Vert\hat y_i - y_i\Vert^2\,/\,\Vert y_i\Vert^2$  per layer",
           "pooled over the batch; 1 = as bad as deleting the FFN", size=15, size2=11)
    cv.arrow([(1262, 622), (1278, 622), (1278, 336), (1262, 336)], color="#2E9E4F", dashed=True, lw=1.4)
    cv.arrow([(1278, 512), (1262, 512)], color="#2E9E4F", dashed=True, lw=1.4)
    cv.text(740, 684, "first 25% of the run's tokens  ·  lr 1e-3  ·  8,192 tok/step", size=12, color="#333333")
    cv.text(740, 708, "one Gemma pass feeds every layer; layers train independently", size=12, color="#333333")
    cv.arrow([(1290, 450), (1378, 450)])
    cv.text(1334, 435, "then", size=12.5, ha="center", color="#333333")
    # panel 3: end to end
    cv.panel(1380, 162, 1960, 728, "3 · END TO END")
    cv.box(1410, 212, 1930, 270, "plain", "training text", "MobileMoE's pre-training mix (Table B.1), 1,024-token sequences", size=14, size2=11)
    cv.arrow([(1535, 270), (1535, 304)])
    cv.arrow([(1805, 270), (1805, 304)])
    cv.box(1410, 306, 1660, 382, "train", "hybrid, transplants on", r"$p_{hyb}$, with grad", size=14, size2=11)
    cv.box(1680, 306, 1930, 382, "gemma", "Gemma, transplants off", r"$p_A$, no grad", size=14, size2=11)
    cv.arrow([(1535, 382), (1535, 416)])
    cv.arrow([(1805, 382), (1805, 416)])
    cv.box(1410, 418, 1930, 488, "embed", "logits in 256-position chunks", r"$256\times262{,}144$ in fp32 = 0.27 GB; backward recomputes them", size=14, size2=11)
    cv.arrow([(1670, 488), (1670, 522)])
    cv.box(1410, 524, 1930, 604, "loss", r"$\mathcal{L}_2 = \mathrm{CE}(y,\,p_{hyb}) + \beta\,\mathrm{KL}(p_A\,\Vert\,p_{hyb})$",
           r"$\beta = 1$: recover Gemma, do not drift", size=15, size2=11)
    cv.arrow([(1408, 564), (1394, 564), (1394, 344), (1408, 344)], color="#2E9E4F", dashed=True, lw=1.4)
    cv.text(1410, 636, r"gradients reach only $W_{in}$, $W_{out}$ (green dashes)", size=12, color="#333333")
    cv.text(1410, 662, "16,384 tokens per step  ·  lr 3e-4, cosine  ·  AdamW", size=12, color="#333333")
    cv.text(1410, 688, "the other 75%: 0.75M tokens in Stage 1, 15M in Stage 2", size=12, color="#333333")
    cv.legend(762, [("train", r"trainable: $W_{in}$, $W_{out}$", False), ("moe", "frozen, from MobileMoE-M", False),
                    ("gemma", "frozen, from Gemma 3 1B", False), ("loss", "loss", False)])
    cv.text(40, 800, "AdamW: betas (0.9, 0.95), eps 1e-8, gradient clipping 1.0 (per layer in local matching). "
                     "Every token is read once. The first position of each sequence is left out of ridge and local matching.",
            size=12, color=FAINT)
    cv.save("fig4_training_pipeline.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
