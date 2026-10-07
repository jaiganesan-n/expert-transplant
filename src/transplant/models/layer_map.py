"""Which Gemma (A) layers get a transplant, and which MobileMoE (B) layer each one borrows."""
from __future__ import annotations

from typing import Iterable


def relative_depth(i: int, n_a: int, n_b: int) -> int:
    """j = round(i * (L_B - 1) / (L_A - 1)). With L_A = L_B = 26 this is j = i."""
    if n_a == 1:
        return 0
    return int(round(i * (n_b - 1) / (n_a - 1)))


def map_layers(a_layers: Iterable[int], n_a: int, n_b: int, rule: str = "relative_depth") -> dict[int, int]:
    """Return {A layer: B layer}.

    Rules (ablation AB4):
      * ``relative_depth`` (default)
      * ``offset:K``   B layer = relative_depth(i) + K, clipped to [0, L_B - 1]
      * ``fixed:J``    every A layer borrows B layer J
    """
    out: dict[int, int] = {}
    for i in a_layers:
        if rule == "relative_depth":
            j = relative_depth(i, n_a, n_b)
        elif rule.startswith("offset:"):
            j = relative_depth(i, n_a, n_b) + int(rule.split(":", 1)[1])
            j = min(max(j, 0), n_b - 1)
        elif rule.startswith("fixed:"):
            j = int(rule.split(":", 1)[1])
        else:
            raise ValueError(f"unknown layer-map rule '{rule}'")
        if not 0 <= j < n_b:
            raise ValueError(f"B layer {j} out of range for L_B={n_b}")
        out[int(i)] = j
    return out


def middle_out(n_layers: int, k: int, center: int | None = None) -> list[int]:
    """k layers added middle-out: center, center-1, center+1, center-2, ...

    For Gemma's 26 layers (center 13): k=4 -> 11..14, k=8 -> 9..16.
    """
    center = n_layers // 2 if center is None else center
    order = [center]
    step = 1
    while len(order) < n_layers:
        for cand in (center - step, center + step):
            if 0 <= cand < n_layers and cand not in order:
                order.append(cand)
        step += 1
    return sorted(order[:k])


def parse_layers(spec, n_layers: int) -> list[int]:
    """Accepts an int, a list, 'all', 'middle:K', 'first:K', 'last:K' or 'k:K' (middle-out)."""
    if isinstance(spec, int):
        layers = [spec]
    elif isinstance(spec, (list, tuple)):
        layers = [int(x) for x in spec]
    elif isinstance(spec, str):
        s = spec.strip()
        if s == "all":
            layers = list(range(n_layers))
        elif s.startswith(("middle:", "k:")):
            layers = middle_out(n_layers, int(s.split(":", 1)[1]))
        elif s.startswith("first:"):
            layers = list(range(int(s.split(":", 1)[1])))
        elif s.startswith("last:"):
            k = int(s.split(":", 1)[1])
            layers = list(range(n_layers - k, n_layers))
        else:
            layers = [int(x) for x in s.replace("[", "").replace("]", "").split(",") if x.strip()]
    else:
        raise ValueError(f"cannot parse layer spec {spec!r}")
    for i in layers:
        if not 0 <= i < n_layers:
            raise ValueError(f"layer {i} out of range for {n_layers} layers")
    return sorted(set(layers))
