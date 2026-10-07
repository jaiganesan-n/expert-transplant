"""Linear CKA between two sets of representations of the same tokens (rows)."""
from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    hsic = (y.T @ x).pow(2).sum()
    norm = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return float(hsic / norm.clamp_min(1e-12))
