"""Checkpoints hold only what training changes: projections, optimizer, schedule, scaler,
data-stream position and RNG state. About 9 MB per layer of fp32 projections, plus
twice that for Adam's moments. A run stopped for any reason (sleep, reboot, Ctrl-C)
resumes exactly with ``--resume``."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch

from ..models import surgery
from ..utils.seed import rng_state, set_rng_state


def ckpt_due(cfg: dict, step: int, last_save: float) -> bool:
    """Save every ``checkpoint.every_steps`` steps or every ``checkpoint.every_minutes``
    minutes, whichever comes first, so a long run never loses more than half an hour."""
    ck = cfg["checkpoint"]
    by_steps = (step + 1) % int(ck.get("every_steps", 200)) == 0
    by_time = (time.time() - last_save) >= 60 * float(ck.get("every_minutes", 30))
    return by_steps or by_time


def save_checkpoint(path: str | Path, *, model, optimizer, scheduler, scaler, step: int, data_state: dict, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "projections": surgery.projection_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "data_state": data_state,
        "rng": rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, scaler=None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    surgery.load_projection_state(model, payload["projections"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    set_rng_state(payload["rng"])
    return payload


def save_projections(path: str | Path, model) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surgery.projection_state(model), path)


def load_projections(path: str | Path, model, strict: bool = True) -> None:
    surgery.load_projection_state(model, torch.load(path, map_location="cpu"), strict=strict)
