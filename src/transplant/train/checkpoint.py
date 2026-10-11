"""Checkpoints hold only what training changes: projections, optimizer, schedule, scaler,
data-stream position and RNG state. About 9 MB per layer of fp32 projections, plus
twice that for Adam's moments. A run stopped for any reason (sleep, reboot, Ctrl-C)
resumes exactly with ``--resume``.

Every write goes to a temporary file that is flushed to the disk before it replaces the
checkpoint, and the checkpoint it replaces is kept as ``<name>.prev.pt``; if the newest file
is damaged (power cut during a write), loading falls back to the previous one. A small
``<name>.json`` beside it records the step and time, for progress views.
"""
from __future__ import annotations

import json
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
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        _flush_to_disk(fh)
    prev = _prev_path(path)
    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)
    _write_json_atomic(path.with_suffix(".json"), {"step": step, "time": time.time()})


def _prev_path(path: Path) -> Path:
    return path.with_name(path.stem + ".prev" + path.suffix)


def _flush_to_disk(fh) -> None:
    fh.flush()
    try:
        import fcntl

        full = getattr(fcntl, "F_FULLFSYNC", None)  # macOS: past the drive's own cache
        if full is not None:
            fcntl.fcntl(fh.fileno(), full)
            return
    except (ImportError, OSError):
        pass
    os.fsync(fh.fileno())


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def checkpoint_candidates(path: str | Path) -> list[Path]:
    """Where a checkpoint may be, newest first: the file, an unfinished replace, the previous one."""
    path = Path(path)
    return [p for p in (path, path.with_suffix(".tmp"), _prev_path(path)) if p.exists()]


def load_latest_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, scaler=None,
                           log=print) -> Optional[dict]:
    """The newest checkpoint that loads, or None when there is none."""
    for cand in checkpoint_candidates(path):
        try:
            payload = load_checkpoint(cand, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        except Exception as e:  # damaged or half-written: try the next one
            log(f"[ckpt] {cand.name} could not be read ({type(e).__name__}); trying an older checkpoint")
            continue
        if cand != Path(path):
            log(f"[ckpt] resumed from {cand.name}")
        return payload
    return None


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
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        torch.save(surgery.projection_state(model), fh)
        _flush_to_disk(fh)
    os.replace(tmp, path)


def load_projections(path: str | Path, model, strict: bool = True) -> None:
    surgery.load_projection_state(model, torch.load(path, map_location="cpu"), strict=strict)
