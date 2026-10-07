"""Gap recovery: the share of the damage from removing Gemma's FFN that the transplant repairs.

    GR = (ppl_ablated - ppl_hybrid) / (ppl_ablated - ppl_A)

GR = 1: back to Gemma. GR > 1: better than Gemma. With many layers removed the ablated
perplexity explodes and GR saturates, so the same ratio in loss space (GR_CE, with
CE = log ppl) and the plain ratio ppl_hybrid / ppl_A are reported beside it, together
with ppl_ablated - ppl_A (a small gap makes GR noisy).
"""
from __future__ import annotations

import math


def gap_recovery(ablated: float, hybrid: float, base: float) -> float:
    gap = ablated - base
    if abs(gap) < 1e-12:
        return float("nan")
    return (ablated - hybrid) / gap


def summarize_ppl(ce_a: float, ce_ablated: float, ce_hybrid: float, acc: dict | None = None) -> dict:
    """``acc``: optional next-token top-1 accuracies {"a", "ablated", "hybrid"}."""
    ppl_a, ppl_abl, ppl_hyb = math.exp(ce_a), math.exp(ce_ablated), math.exp(ce_hybrid)
    out = {
        "ppl_a": ppl_a,
        "ppl_ablated": ppl_abl,
        "ppl_hybrid": ppl_hyb,
        "ce_a": ce_a,
        "ce_ablated": ce_ablated,
        "ce_hybrid": ce_hybrid,
        "gap_ablated_minus_a": ppl_abl - ppl_a,
        "gr": gap_recovery(ppl_abl, ppl_hyb, ppl_a),
        "gr_ce": gap_recovery(ce_ablated, ce_hybrid, ce_a),
        "ppl_ratio": ppl_hyb / ppl_a,
    }
    if acc:
        out.update({f"acc_{k}": v for k, v in acc.items()})
    return out
