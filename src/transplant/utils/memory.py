"""Peak-memory helpers (measured, not estimated: plan assumption A9).

CUDA keeps its own peak counter. MPS (Apple GPU) has none, so two peaks are taken at the
points where training calls ``track_peak``:

* driver: everything Metal holds for this process, including blocks PyTorch has freed but
  keeps cached. This is what competes with macOS for unified memory, and what the limits use.
* live: memory held by tensors that still exist.

All values are GiB.
"""
from __future__ import annotations

from typing import Optional

import torch

GIB = 1024**3
_MPS_PEAK = 0.0
_MPS_PEAK_LIVE = 0.0


def reset_peak(device: torch.device) -> None:
    global _MPS_PEAK, _MPS_PEAK_LIVE
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        _MPS_PEAK = _MPS_PEAK_LIVE = 0.0
        track_peak(device)


def track_peak(device: torch.device) -> None:
    global _MPS_PEAK, _MPS_PEAK_LIVE
    if device.type == "mps":
        try:
            _MPS_PEAK = max(_MPS_PEAK, torch.mps.driver_allocated_memory() / GIB)
            _MPS_PEAK_LIVE = max(_MPS_PEAK_LIVE, torch.mps.current_allocated_memory() / GIB)
        except Exception:  # pragma: no cover - older torch
            pass


def live_gb(device: torch.device) -> float:
    """Memory held by live tensors right now."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / GIB
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / GIB
    return float("nan")


def peak_live_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / GIB
    if device.type == "mps":
        track_peak(device)
        return _MPS_PEAK_LIVE
    return float("nan")


def recommended_max_gb(device: torch.device) -> Optional[float]:
    """Metal's recommendedMaxWorkingSetSize: Apple's estimate of how much memory the GPU can
    use on this Mac without hurting performance (beyond it macOS compresses and swaps).
    On CUDA, the card's total memory."""
    try:
        if device.type == "mps":
            return torch.mps.recommended_max_memory() / GIB
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).total_memory / GIB
    except Exception:  # pragma: no cover - older torch
        return None
    return None


def peak_gb(device: torch.device) -> float:
    """Driver peak on MPS (see the module docstring); allocator peak on CUDA."""
    if device.type == "cuda":
        return torch.cuda.max_memory_reserved(device) / GIB
    if device.type == "mps":
        track_peak(device)
        return _MPS_PEAK
    return float("nan")


def peak_text(device: torch.device) -> str:
    """", peak 7.9 GB" for log lines, or nothing on the CPU (no counter there)."""
    gb = peak_gb(device)
    return "" if gb != gb else f", peak {gb:.1f} GB"


def process_gb() -> Optional[float]:
    """Resident memory of this process (CPU side; on a Mac the GPU's share is reported apart)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / GIB
    except Exception:  # pragma: no cover - psutil missing
        return None


def memory_text(device: torch.device) -> str:
    """'GPU 8.1 GB, process 2.3 GB' for progress lines."""
    parts = []
    if device.type == "mps":
        parts.append(f"GPU {torch.mps.driver_allocated_memory() / GIB:.1f} GB")
    elif device.type == "cuda":
        parts.append(f"GPU {torch.cuda.memory_reserved(device) / GIB:.1f} GB")
    proc = process_gb()
    if proc is not None:
        parts.append(f"process {proc:.1f} GB")
    return ", ".join(parts) or "memory n/a"


def empty_cache(device: torch.device) -> None:
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
