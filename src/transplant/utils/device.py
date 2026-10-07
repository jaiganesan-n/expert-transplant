"""Device and precision selection for Apple MPS (the M5 this project trains on), CUDA and CPU."""
from __future__ import annotations

from dataclasses import dataclass

import torch

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8  # Ampere or newer


@dataclass
class PrecisionPolicy:
    """Where each part of the hybrid runs and in which dtype. Never below 16-bit."""

    a_dtype: torch.dtype  # Gemma: weights and residual stream
    b_dtype: torch.dtype  # MobileMoE MoE blocks (frozen)
    proj_dtype: torch.dtype  # W_in / W_out master weights
    grad_scaler: bool  # dynamic loss scaling when gradients cross fp16 modules

    def describe(self) -> dict:
        return {
            "a_dtype": str(self.a_dtype).replace("torch.", ""),
            "b_dtype": str(self.b_dtype).replace("torch.", ""),
            "proj_dtype": str(self.proj_dtype).replace("torch.", ""),
            "grad_scaler": self.grad_scaler,
        }


def resolve_precision(cfg_precision: dict, device: torch.device) -> PrecisionPolicy:
    """Turn the ``precision`` config block into dtypes for this device.

    ``auto`` picks:
      * CUDA without bf16 (pre-Ampere): Gemma in fp32 (its activations overflow fp16),
        MobileMoE blocks in fp16, fp32 projections, loss scaling on.
      * CUDA with bf16: everything bf16 except fp32 projections.
      * MPS (Mac): bf16 for both models (macOS 14+), fp32 projections.
      * CPU: fp32 everywhere.
    """
    mode = cfg_precision.get("mode", "auto")
    if mode == "auto":
        if device.type == "cuda":
            if cuda_supports_bf16():
                a, b, scaler = "bfloat16", "bfloat16", False
            else:
                a, b, scaler = "float32", "float16", True
        elif device.type == "mps":
            a, b, scaler = "bfloat16", "bfloat16", False
        else:
            a, b, scaler = "float32", "float32", False
        p = "float32"
    else:
        a = cfg_precision["a_dtype"]
        b = cfg_precision["b_dtype"]
        p = cfg_precision.get("proj_dtype", "float32")
        scaler = bool(cfg_precision.get("grad_scaler", DTYPES[b] == torch.float16))
    policy = PrecisionPolicy(DTYPES[a], DTYPES[b], DTYPES[p], scaler and device.type == "cuda")
    for dt in (policy.a_dtype, policy.b_dtype, policy.proj_dtype):
        if dt not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("precision below 16-bit is not allowed in this project")
    return policy
