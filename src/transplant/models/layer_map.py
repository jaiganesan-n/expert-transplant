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


def spread_layers(n_a: int, n_b: int) -> dict[int, int]:
    """Layout ``spread`` (Stage 4): every B layer once, at its relative depth in A.

    B layer j goes to A layer round(j * (L_A - 1) / (L_B - 1)); the A layers no B layer reaches keep
    their own FFN. Gemma 3 1B (26) with MobileMoE-S (20): A layers 0, 1, 3, 4, 5, 7, ..., 24, 25 get
    B layers 0..19 and the FFNs of layers 2, 6, 10, 15, 19 and 23 stay. Returns {A layer: B layer}.
    """
    if n_b > n_a:
        raise ValueError(f"spread needs L_B <= L_A (got L_B={n_b}, L_A={n_a})")
    out = {relative_depth(j, n_b, n_a): j for j in range(n_b)}
    if len(out) != n_b:
        raise ValueError(f"spread: two B layers fall on the same A layer for L_A={n_a}, L_B={n_b}")
    return dict(sorted(out.items()))


def resolve_layer_map(pair: dict, n_a: int, n_b: int) -> tuple[list[int], dict[int, int]]:
    """(transplanted A layers, {A layer: B layer}) from ``pair.layers`` and ``pair.layer_map``.

    ``pair.layers: spread`` uses every B layer once at its relative depth (``spread_layers``);
    anything else is a layer spec for ``parse_layers`` mapped with ``pair.layer_map``.
    """
    spec = pair.get("layers", "all")
    if isinstance(spec, str) and spec.strip() == "spread":
        lm = spread_layers(n_a, n_b)
        return sorted(lm), lm
    a_layers = parse_layers(spec, n_a)
    return a_layers, map_layers(a_layers, n_a, n_b, pair.get("layer_map", "relative_depth"))


def kept_layers(n_a: int, transplanted: list[int]) -> list[int]:
    """The A layers that keep their own FFN."""
    return [i for i in range(n_a) if i not in set(transplanted)]
