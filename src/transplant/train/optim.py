"""Optimizer and schedules.

AdamW, the optimizer of the dense models in this plan (Gemma 3's own report names no
optimizer): betas (0.9, 0.95), eps 1e-8, gradient clipping at 1.0. Weight decay is 0
in local matching and 0.01 on the projections end to end.
"""
from __future__ import annotations

import math

import torch


def make_optimizer(params, lr: float, weight_decay: float, optim_cfg: dict) -> torch.optim.Optimizer:
    name = optim_cfg.get("name", "adamw").lower()
    if name != "adamw":
        raise ValueError("this project uses AdamW (optim.name: adamw)")
    betas = tuple(float(b) for b in optim_cfg.get("betas", (0.9, 0.95)))
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=weight_decay,
        foreach=True,
    )


def lr_lambda(schedule: str, total_steps: int, warmup_steps: int, min_lr_frac: float = 0.1):
    def constant(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        return 1.0

    def cosine(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    if schedule == "constant":
        return constant
    if schedule == "cosine":
        return cosine
    raise ValueError(f"unknown schedule {schedule}")


def make_scheduler(opt, schedule: str, total_steps: int, warmup_steps: int, min_lr_frac: float = 0.1):
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(schedule, total_steps, warmup_steps, min_lr_frac))


def steps_for_tokens(tokens: int, micro_batch: int, grad_accum: int, seq_len: int) -> int:
    """Whole optimizer steps that fit in the budget (rounded down, at least 1), so a run
    never reads more than its token budget."""
    return max(1, int(tokens) // (micro_batch * grad_accum * seq_len))
