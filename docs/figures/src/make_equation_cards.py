"""Render the equation cards (PNG, 4,000 px wide) with LaTeX into docs/figures/equations/.

    python docs/figures/src/make_equation_cards.py      # needs pdflatex and pdftoppm

The .tex source of every card is written next to this script (equations/*.tex).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "equations"
SRC = HERE / "equations"

CARDS = [
    ("eq01_gemma_layer", "Gemma's decoder layer and the FFN slot",
     r"""h = X + P_{\mathrm{att}}\big(\mathrm{Attn}(N_{\mathrm{in}}(X))\big), \qquad
X' = h + P_A\Big(\underbrace{\mathrm{FFN}_A(N_A(h))}_{\text{the slot the transplant takes}}\Big)"""),
    ("eq02_replace", "Replace: one transplanted layer",
     r"""h^{\mathrm{out}}_t = h_t + P_A\Big(\underbrace{W_{out}}_{1152\times1024}\;\mathrm{MoE}_B\big(N_B(\underbrace{W_{in}}_{1024\times1152}\,x_t)\big)\Big),
\qquad x_t = N_A(h_t) \in \mathbb{R}^{1152}"""),
    ("eq03_router", "MobileMoE's router: sigmoid scores, biased top-4, normalized and scaled",
     r"""\hat u_t = N_B(u_t), \qquad s_t = \sigma(W_r\,\hat u_t) \in (0,1)^{60}, \qquad \mathcal{T}_t = \mathrm{TopK}(s_t + b,\,4)
\\[6pt]
g_{t,e} = 2.5\,\frac{s_{t,e}}{\sum_{e' \in \mathcal{T}_t} s_{t,e'}}, \quad e \in \mathcal{T}_t, \qquad \sum_{e \in \mathcal{T}_t} g_{t,e} = 2.5"""),
    ("eq04_moe_output", "The MoE block's output",
     r"""m_t = \underbrace{S(\hat u_t)}_{\substack{\text{shared, always on}\\ 1024\to2048\to1024}}
+ \sum_{e \in \mathcal{T}_t} g_{t,e}\;\underbrace{E_e(\hat u_t)}_{\substack{\text{4 of 60 routed}\\ 1024\to512\to1024}}"""),
    ("eq05_projection", "W_in: a change of coordinates",
     r"""u_t = W_{in}\,x_t, \qquad u_{t,r} = \sum_{c=0}^{1151} W_{in}[r,c]\;x_{t,c}, \quad r = 0,\dots,1023"""),
    ("eq06_rmsnorm_scale", "RMSNorm ignores scale",
     r"""N(z) = \frac{z}{\sqrt{\frac{1}{d}\sum_{c=0}^{d-1} z_c^{2} + \epsilon}} \odot \gamma
\\[6pt]
N_B(\alpha\,W_{in}x_t) \approx N_B(W_{in}x_t), \qquad P_A(\alpha\,W_{out}m_t) \approx P_A(W_{out}m_t), \qquad \alpha > 0"""),
    ("eq07_layer_map", "Layer map by relative depth",
     r"""j = \mathrm{round}\Big(i\cdot\frac{L_B-1}{L_A-1}\Big)
\\[6pt]
\text{primary: } \frac{26-1}{26-1} = 1 \;\Rightarrow\; j = i, \qquad
\text{fallback: } \mathrm{round}\Big(20\cdot\frac{31}{39}\Big) = \mathrm{round}(15.90) = 16"""),
    ("eq08_ridge", "Ridge initialization from streamed statistics",
     r"""\hat W = \arg\min_{W}\,\lVert XW - Y\rVert_F^{2} + \lambda\lVert W\rVert_F^{2} = (X^{\top}X + \lambda I)^{-1}X^{\top}Y,
\qquad W_{in} = \hat W^{\top}
\\[6pt]
X^{\top}X = \sum_{b} X_b^{\top}X_b \in \mathbb{R}^{1152\times1152}, \qquad X^{\top}Y = \sum_{b} X_b^{\top}Y_b \in \mathbb{R}^{1152\times1024},
\qquad Y_t = \frac{h^{B}_t}{\mathrm{rms}(h^{B}_t)}"""),
    ("eq09_local_matching", "Local matching: each layer imitates Gemma's FFN",
     r"""\mathcal{L}_1 = \frac{\sum_t \lVert \hat y_t - y_t\rVert^{2}}{\sum_t \lVert y_t\rVert^{2}}
= \frac{\sum_t \sum_{c=0}^{1151} (\hat y_{t,c} - y_{t,c})^{2}}{\sum_t \sum_{c=0}^{1151} y_{t,c}^{2}}
\\[6pt]
\hat y_t = W_{out}\,\mathrm{MoE}_B\big(N_B(W_{in}x_t)\big), \qquad y_t = \mathrm{FFN}_A(x_t), \qquad \hat y = 0 \;\Rightarrow\; \mathcal{L}_1 = 1"""),
    ("eq10_end_to_end", "End to end: next-token loss plus KL to Gemma",
     r"""\mathcal{L}_2 = \mathrm{CE}(y,\,p_{hyb}) + \beta\,\mathrm{KL}(p_A \,\Vert\, p_{hyb}), \qquad \beta = 1
\\[6pt]
= -\frac{1}{n}\sum_{t} \log p_{hyb}(y_t \mid y_{<t})
+ \frac{\beta}{n}\sum_{t}\sum_{v=0}^{262143} p_A(v \mid y_{<t})\,\log\frac{p_A(v \mid y_{<t})}{p_{hyb}(v \mid y_{<t})}"""),
    ("eq11_gap_recovery", "Gap recovery",
     r"""\mathrm{GR} = \frac{\mathrm{ppl}_{\mathrm{ablated}} - \mathrm{ppl}_{\mathrm{hybrid}}}{\mathrm{ppl}_{\mathrm{ablated}} - \mathrm{ppl}_{A}},
\qquad
\mathrm{GR}_{\mathrm{CE}} = \frac{\mathrm{CE}_{\mathrm{ablated}} - \mathrm{CE}_{\mathrm{hybrid}}}{\mathrm{CE}_{\mathrm{ablated}} - \mathrm{CE}_{A}},
\qquad \mathrm{CE} = \log \mathrm{ppl}
\\[6pt]
\text{example: } \mathrm{ppl}_A = 12.0,\ \mathrm{ppl}_{\mathrm{ablated}} = 14.0,\ \mathrm{ppl}_{\mathrm{hybrid}} = 12.2
\;\Rightarrow\; \mathrm{GR} = \frac{1.8}{2.0} = 0.9"""),
    ("eq12_routing_health", "Routing health and agreement",
     r"""\tilde H = -\frac{1}{\log 60}\sum_{e=0}^{59} f_e \log f_e \;\ge\; 0.7 \;\iff\; e^{H} \ge 60^{0.7} \approx 17.6 \text{ experts in effective use}
\\[6pt]
J_t = \frac{\lvert \mathcal{T}^{\mathrm{hyb}}_t \cap \mathcal{T}^{B}_t\rvert}{\lvert \mathcal{T}^{\mathrm{hyb}}_t \cup \mathcal{T}^{B}_t\rvert},
\qquad \mathbb{E}[J]_{\text{chance}} = \sum_{k=0}^{4} \frac{\binom{4}{k}\binom{56}{4-k}}{\binom{60}{4}}\cdot\frac{k}{8-k} \approx 0.039"""),
    ("eq13_parameter_count", "Counting parameters",
     r"""\underbrace{2\cdot 1152\cdot 1024}_{W_{in},\;W_{out}} = 2{,}359{,}296 \text{ per layer}, \quad \times 26 = 61{,}341{,}696 \text{ trainable}
\\[6pt]
\underbrace{4\cdot 3\cdot 1024\cdot 512}_{\text{4 routed experts}} + \underbrace{3\cdot 1024\cdot 2048}_{\text{shared expert}}
+ \underbrace{60\cdot 1024}_{\text{router}} = 12{,}644{,}352 \text{ active per token per layer}
\\[6pt]
12{,}644{,}352 + 2{,}359{,}296 - \underbrace{23{,}887{,}872}_{\mathrm{FFN}_A\ \text{removed}} = -8{,}884{,}224 \text{ active per token per transplanted layer}"""),
]

TEMPLATE = r"""\documentclass[border=40pt]{standalone}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{xcolor}
\usepackage[scaled=0.95]{helvet}
\definecolor{bg}{HTML}{F8F8F8}
\definecolor{muted}{HTML}{6B7280}
\pagecolor{bg}
\begin{document}
\begin{minipage}{26cm}
{\sffamily\bfseries\Large\color{muted} %(title)s}\par\vspace{-14pt}
{\LARGE
\begin{gather*}
%(body)s
\end{gather*}}
\end{minipage}
\end{document}
"""


def render(name: str, title: str, body: str, workdir: Path) -> Path:
    tex = TEMPLATE % {"title": title.replace("W_in", r"$W_{in}$"), "body": body}
    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / f"{name}.tex").write_text(tex)
    (workdir / f"{name}.tex").write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=workdir,
                   check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "4000", "-scale-to-y", "-1", f"{name}.pdf",
                    str(OUT / name)], cwd=workdir, check=True)
    return OUT / f"{name}.png"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    try:
        for name, title, body in CARDS:
            print("wrote", render(name, title, body, work))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
