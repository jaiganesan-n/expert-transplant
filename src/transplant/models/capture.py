"""Forward hooks that read activations out of A and B, and stop a forward pass early."""
from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import torch
from torch import nn

from .surgery import decoder_layers


class StopForward(Exception):
    """Raised by a hook once every needed activation is captured."""


@contextlib.contextmanager
def capture_ffn_io(model_a: nn.Module, layers: Iterable[int], want_output: bool = True, stop_after: Optional[int] = None):
    """Capture x = N_A(h) (the FFN-slot input) and, optionally, the slot output, per layer.

    Yields a dict {layer: {"x": tensor, "y": tensor}}. If ``stop_after`` is set, the
    forward pass stops right after that layer's FFN slot (wrap the call in ``run_until``).
    """
    store: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    dec = decoder_layers(model_a)
    for i in sorted(set(layers)):
        def hook(module, inputs, output, i=i):
            entry = store.setdefault(i, {})
            entry["x"] = inputs[0].detach()
            if want_output:
                entry["y"] = output.detach()
            if stop_after is not None and i == stop_after:
                raise StopForward

        handles.append(dec[i].mlp.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def capture_module_inputs(modules: dict[int, nn.Module], stop_after: Optional[int] = None):
    """Capture the first input of each module (e.g. B's pre-MoE norm: B's native MoE-block input)."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    for j, mod in modules.items():
        def pre_hook(module, inputs, j=j):
            store[j] = inputs[0].detach()
            if stop_after is not None and j == stop_after:
                raise StopForward

        handles.append(mod.register_forward_pre_hook(pre_hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def run_until(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StopForward:
        return None
