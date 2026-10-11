"""Hero banner for the README (Project J'AI · Expert Transplant), light and dark versions.

    python docs/figures/src/make_banner.py      # needs playwright with Chromium; writes docs/figures/banner_{light,dark}.png

Left: Gemma 3 1B (host), every FFN slot either transplanted (orange MoE between two green projections) or kept
(blue). Right: MobileMoE (donor), whose MoE blocks are borrowed. Centre: one transplanted slot, zoomed in.
Colours follow the architecture figures: green trainable, orange frozen MobileMoE, blue frozen Gemma.
"""
from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
W, H = 1600, 560

THEMES = {
    "light": dict(bg="#F8F8F8", edge="#E1E0D9", panel="#FFFFFF", panel_edge="#D5D8DC", ink="#222222", muted="#6B7280",
                  faint="#9AA0A8", gemma=("#DCE6F2", "#8FAAD6"), attn=("#FFF3B0", "#D9B92B"), moe=("#FDE3C8", "#E8892B"),
                  lit="#E8892B", train=("#D5F0DD", "#2E9E4F"), norm=("#E8E3F5", "#A08ED6"), off=("#F2F2F2", "#C4C4C4"),
                  arrow="#3A3F47"),
    "dark": dict(bg="#161B22", edge="#30363D", panel="#0D1117", panel_edge="#30363D", ink="#E6EDF3", muted="#8B949E",
                 faint="#6E7681", gemma=("#1C2B3F", "#5B8FD6"), attn=("#3A3214", "#D9B92B"), moe=("#3D2814", "#E8892B"),
                 lit="#F0A050", train=("#13301D", "#3FB950"), norm=("#2A2440", "#A08ED6"), off=("#21262D", "#484F58"),
                 arrow="#C9D1D9"),
}


def rect(x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} {extra}/>'


def text(x, y, s, size, fill, weight=400, anchor="start", style="", family="Inter"):
    return (f'<text x="{x}" y="{y}" font-family="{family}, DejaVu Sans, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')


def experts(x, y, cols, rows, size, gap, fill, stroke, lit, lit_cells=()):
    out = []
    for r in range(rows):
        for c in range(cols):
            on = (r, c) in lit_cells
            out.append(rect(x + c * (size + gap), y + r * (size + gap), size, size,
                            lit if on else fill, lit if on else stroke, rx=2.5, sw=1.1))
    return "".join(out)


def svg(t: dict) -> str:
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    p.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
             '<marker id="ahm" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{t["moe"][1]}"/></marker></defs>')
    p.append(rect(1, 1, W - 2, H - 2, t["bg"], t["edge"], rx=26, sw=1.5))

    # ---- left: Gemma 3 1B, the host
    gx, gw, top, rh, gap = 70, 360, 118, 40, 11
    p.append(text(gx, 80, "Gemma 3 1B", 32, t["ink"], 700))
    p.append(text(gx + 208, 80, "host · frozen", 20, t["muted"]))
    pattern = ["T", "T", "K", "T", "T", "K", "T"]
    zoom_row = 3
    for i, kind in enumerate(pattern):
        y = top + i * (rh + gap)
        p.append(rect(gx, y, gw, rh, t["panel"], t["panel_edge"], rx=9, sw=1.2))
        p.append(rect(gx + 10, y + 8, 74, rh - 16, *t["attn"], rx=6, sw=1.2))
        p.append(text(gx + 47, y + 26, "Attn", 13, t["ink"], 500, "middle"))
        sx = gx + 98
        if kind == "T":
            p.append(rect(sx, y + 7, 14, rh - 14, *t["train"], rx=3, sw=1.3))
            p.append(rect(sx + 20, y + 6, 202, rh - 12, *t["moe"], rx=6, sw=1.3))
            p.append(experts(sx + 30, y + 12, 12, 2, 6.5, 9.2, t["panel"], t["moe"][1], t["lit"],
                             lit_cells={(0, 2), (1, 5), (0, 8), (1, 10)}))
            p.append(rect(sx + 228, y + 7, 14, rh - 14, *t["train"], rx=3, sw=1.3))
        else:
            p.append(rect(sx, y + 6, 242, rh - 12, *t["gemma"], rx=6, sw=1.3))
            p.append(text(sx + 121, y + 26, "Gemma's own FFN (kept)", 13, t["ink"], 500, "middle"))
    bottom = top + len(pattern) * (rh + gap) - gap
    p.append(text(gx, bottom + 36, "every FFN slot: transplanted, or kept", 18, t["muted"]))

    # ---- right: MobileMoE, the donor
    dx = 1170
    p.append(text(dx, 80, "MobileMoE", 32, t["ink"], 700))
    p.append(text(dx + 192, 80, "donor · frozen", 20, t["muted"]))
    for i in range(len(pattern)):
        y = top + i * (rh + gap)
        p.append(rect(dx, y, gw, rh, t["panel"], t["panel_edge"], rx=9, sw=1.2))
        p.append(rect(dx + 10, y + 8, 74, rh - 16, *t["off"], rx=6, sw=1.2, dash="4 3"))
        p.append(text(dx + 47, y + 26, "Attn", 13, t["faint"], 500, "middle"))
        p.append(rect(dx + 98, y + 6, 252, rh - 12, *t["moe"], rx=6, sw=1.3))
        p.append(experts(dx + 110, y + 12, 14, 2, 6.5, 9.2, t["panel"], t["moe"][1], t["lit"],
                         lit_cells={(0, 1), (1, 4), (0, 9), (1, 12)}))
    p.append(text(dx, bottom + 36, "only its MoE blocks are borrowed", 18, t["muted"]))

    # ---- centre: one transplanted slot, zoomed in
    cx, cy, cw, ch = 478, 150, 646, 268
    zy = top + zoom_row * (rh + gap)
    p.append(f'<path d="M {gx + gw - 8} {zy + 2} L {cx} {cy}" stroke="{t["faint"]}" stroke-width="1.2" fill="none" stroke-dasharray="3 4"/>')
    p.append(f'<path d="M {gx + gw - 8} {zy + rh - 2} L {cx} {cy + ch}" stroke="{t["faint"]}" stroke-width="1.2" fill="none" stroke-dasharray="3 4"/>')
    p.append(rect(cx, cy, cw, ch, t["panel"], t["panel_edge"], rx=16, sw=1.5))
    p.append(text(cx + 22, cy + 36, "one transplanted FFN slot", 18, t["muted"], 600))

    my = cy + 140  # flow axis
    def arrow(x1, x2, y=my):
        return f'<path d="M {x1} {y} L {x2} {y}" stroke="{t["arrow"]}" stroke-width="1.8" fill="none" marker-end="url(#ah)"/>'
    p.append(text(cx + 30, my + 7, "x", 24, t["ink"], 500, "middle", 'font-style="italic"'))
    p.append(arrow(cx + 44, cx + 72))
    p.append(rect(cx + 74, my - 36, 86, 72, *t["train"], rx=10, sw=1.8))
    p.append(text(cx + 117, my + 2, "W", 22, t["ink"], 700, "middle"))
    p.append(text(cx + 132, my + 9, "in", 13, t["ink"], 700, "start"))
    p.append(text(cx + 117, my + 24, "trained", 13, t["train"][1], 600, "middle"))
    p.append(arrow(cx + 160, cx + 186))
    p.append(rect(cx + 188, my - 24, 52, 48, *t["norm"], rx=8, sw=1.5))
    p.append(text(cx + 214, my + 6, "N", 18, t["ink"], 600, "middle"))
    p.append(arrow(cx + 240, cx + 266))
    mx, mw, mh = cx + 268, 210, 172
    p.append(rect(mx, my - mh / 2, mw, mh, *t["moe"], rx=12, sw=1.8))
    p.append(text(mx + mw / 2, my - mh / 2 + 24, "router · top 4 of 60", 15, t["ink"], 600, "middle"))
    p.append(experts(mx + 21, my - mh / 2 + 38, 10, 6, 11, 6.2, t["panel"], t["moe"][1], t["lit"],
                     lit_cells={(0, 3), (2, 7), (3, 1), (5, 5)}))
    p.append(text(mx + mw / 2, my + mh / 2 - 12, "+ shared expert · frozen", 14, t["ink"], 500, "middle"))
    p.append(arrow(mx + mw, mx + mw + 26))
    ox = mx + mw + 28
    p.append(rect(ox, my - 36, 86, 72, *t["train"], rx=10, sw=1.8))
    p.append(text(ox + 40, my + 2, "W", 22, t["ink"], 700, "middle"))
    p.append(text(ox + 55, my + 9, "out", 13, t["ink"], 700, "start"))
    p.append(text(ox + 43, my + 24, "trained", 13, t["train"][1], 600, "middle"))
    p.append(arrow(ox + 86, ox + 112))
    p.append(text(ox + 126, my + 7, "y", 24, t["ink"], 500, "middle", 'font-style="italic"'))
    p.append(text(cx + cw / 2, cy + ch - 18, "y = W<tspan dy=\"4\" font-size=\"13\">out</tspan><tspan dy=\"-4\"> · MoE( N( W</tspan>"
                  "<tspan dy=\"4\" font-size=\"13\">in</tspan><tspan dy=\"-4\"> · x ) )</tspan>", 21, t["ink"], 500, "middle"))

    # donor block -> slot
    p.append(f'<path d="M {dx + 4} {top + rh / 2 + (rh + gap)} C {dx - 60} {top - 30}, {mx + mw + 40} {cy - 70}, {mx + mw / 2 + 30} {my - mh / 2 - 4}" '
             f'stroke="{t["moe"][1]}" stroke-width="2" fill="none" stroke-dasharray="7 5" marker-end="url(#ahm)"/>')
    p.append(text(dx - 20, cy - 52, "borrowed, frozen", 17, t["moe"][1], 600, "end"))

    # ---- legend
    ly = H - 28
    items = [(t["train"], "trained: W_in, W_out (both models frozen)"), (t["moe"], "MobileMoE MoE block"),
             (t["gemma"], "Gemma"), (t["norm"], "MobileMoE norm")]
    x = 70
    for (fill, stroke), label in items:
        p.append(rect(x, ly - 15, 20, 18, fill, stroke, rx=4, sw=1.4))
        p.append(text(x + 28, ly, label, 18, t["muted"]))
        x += 28 + len(label) * 8.7 + 42
    p.append("</svg>")
    return "".join(p)


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        for name, theme in THEMES.items():
            page.set_content(f'<html><body style="margin:0;background:transparent">{svg(theme)}</body></html>')
            page.wait_for_timeout(150)
            out = OUT / f"banner_{name}.png"
            page.locator("svg").screenshot(path=str(out), omit_background=True)
            print("wrote", out)
        browser.close()


if __name__ == "__main__":
    main()
