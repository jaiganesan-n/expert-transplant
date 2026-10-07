"""Trainable projections W_in (A's width -> B's width) and W_out (B's width -> A's width).

Forms (ablation AB3): ``linear`` (default), ``lowrank`` (rank r), ``mlp`` (2 layers, GELU).
"""
from __future__ import annotations

import torch
from torch import nn


class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    @torch.no_grad()
    def load_matrix(self, w: torch.Tensor) -> None:
        """Best rank-r approximation of a full (d_out, d_in) matrix (truncated SVD)."""
        u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
        r = self.down.out_features
        root = s[:r].sqrt()
        self.up.weight.copy_((u[:, :r] * root).to(self.up.weight.dtype))
        self.down.weight.copy_((root[:, None] * vh[:r]).to(self.down.weight.dtype))


class TwoLayerMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def make_projection(kind: str, d_in: int, d_out: int, *, rank: int = 256, hidden: int = 2048) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_in, d_out, bias=False)
    if kind == "lowrank":
        return LowRank(d_in, d_out, rank)
    if kind == "mlp":
        return TwoLayerMLP(d_in, d_out, hidden)
    raise ValueError(f"unknown projection kind '{kind}'")


@torch.no_grad()
def load_projection_matrix(proj: nn.Module, w: torch.Tensor) -> bool:
    """Copy a (d_out, d_in) matrix into a projection. Returns False when the form
    cannot take a matrix (the MLP keeps its random init)."""
    if isinstance(proj, nn.Linear):
        proj.weight.copy_(w.to(proj.weight.dtype))
        return True
    if isinstance(proj, LowRank):
        proj.load_matrix(w)
        return True
    return False
