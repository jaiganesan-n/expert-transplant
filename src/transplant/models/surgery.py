"""Model surgery: swap Gemma's FFN for B's MoE block between Gemma's sandwich norms.

Gemma 3's decoder layer computes, after attention,

    h_out = h + P_A( FFN_A( N_A(h) ) )

with N_A = ``pre_feedforward_layernorm`` and P_A = ``post_feedforward_layernorm``.
The transplant replaces only the module in the FFN slot (``layer.mlp``), so both of
Gemma's norms stay where they are:

    h_out = h + P_A( W_out MoE_B( N_B( W_in N_A(h) ) ) )

Only W_in and W_out train. Each wrapper keeps a frozen reference to Gemma's own FFN.
It is used for local-matching targets, the KL teacher (``mode = "original"``) and nothing
else. ``mode = "zero"`` gives the ablated baseline (control C4).
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import torch
from torch import nn

MODES = ("transplant", "original", "zero")


class ReplaceFFN(nn.Module):
    """The FFN slot of one Gemma layer after surgery."""

    def __init__(
        self,
        ffn_a: nn.Module,
        inner: nn.Module,
        w_in: nn.Module,
        w_out: nn.Module,
        *,
        layer_a: int,
        layer_b: int,
        b_dtype: torch.dtype,
        proj_dtype: torch.dtype,
    ):
        super().__init__()
        self.ffn_a = ffn_a
        self.inner = inner
        self.w_in = w_in
        self.w_out = w_out
        self.layer_a = layer_a
        self.layer_b = layer_b
        self.b_dtype = b_dtype
        self.proj_dtype = proj_dtype
        self.mode = "transplant"

    def transplant(self, x: torch.Tensor) -> torch.Tensor:
        u = self.w_in(x.to(self.proj_dtype))
        m = self.inner(u.to(self.b_dtype))
        y = self.w_out(m.to(self.proj_dtype))
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transplant":
            return self.transplant(x)
        if self.mode == "original":
            return self.ffn_a(x)
        if self.mode == "zero":
            return torch.zeros_like(x)
        raise ValueError(f"unknown mode {self.mode}")

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in list(self.w_in.parameters()) + list(self.w_out.parameters()):
            if p.requires_grad:
                yield p
        for p in self.inner.parameters():  # only for AB5 (router-only training)
            if p.requires_grad:
                yield p


class BottleneckFFN(nn.Module):
    """Control C6: Gemma's own FFN behind a trained 1152 -> 1024 -> 1152 linear bottleneck.

    forward(x) = FFN_A( W_out W_in x ). Answers: is the 1024-wide bottleneck itself the limit?
    """

    def __init__(self, ffn_a: nn.Module, w_in: nn.Module, w_out: nn.Module, *, layer_a: int, proj_dtype: torch.dtype):
        super().__init__()
        self.ffn_a = ffn_a
        self.w_in = w_in
        self.w_out = w_out
        self.layer_a = layer_a
        self.layer_b = -1
        self.inner = nn.Identity()
        self.proj_dtype = proj_dtype
        self.mode = "transplant"

    def transplant(self, x: torch.Tensor) -> torch.Tensor:
        z = self.w_out(self.w_in(x.to(self.proj_dtype)))
        return self.ffn_a(z.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transplant":
            return self.transplant(x)
        if self.mode == "original":
            return self.ffn_a(x)
        if self.mode == "zero":
            return torch.zeros_like(x)
        raise ValueError(f"unknown mode {self.mode}")

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in list(self.w_in.parameters()) + list(self.w_out.parameters()):
            if p.requires_grad:
                yield p


SLOT_TYPES = (ReplaceFFN, BottleneckFFN)


def decoder_layers(model_a: nn.Module) -> nn.ModuleList:
    """Gemma3ForCausalLM -> model.model.layers."""
    return model_a.model.layers


def attach(model_a: nn.Module, slots: dict[int, nn.Module]) -> None:
    """Put each wrapper into its Gemma layer's FFN slot (``layer.mlp``)."""
    layers = decoder_layers(model_a)
    for i, slot in slots.items():
        if not isinstance(slot, SLOT_TYPES):
            raise TypeError("slot must be ReplaceFFN or BottleneckFFN")
        if isinstance(layers[i].mlp, SLOT_TYPES):
            raise RuntimeError(f"layer {i} already has a transplant")
        layers[i].mlp = slot


def detach(model_a: nn.Module) -> None:
    """Undo surgery: put Gemma's own FFNs back."""
    for layer in decoder_layers(model_a):
        if isinstance(layer.mlp, SLOT_TYPES):
            layer.mlp = layer.mlp.ffn_a


def slots(model_a: nn.Module) -> dict[int, nn.Module]:
    return {i: layer.mlp for i, layer in enumerate(decoder_layers(model_a)) if isinstance(layer.mlp, SLOT_TYPES)}


def set_mode(model_a: nn.Module, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for slot in slots(model_a).values():
        slot.mode = mode


@contextlib.contextmanager
def mode(model_a: nn.Module, which: str):
    previous = {i: s.mode for i, s in slots(model_a).items()}
    set_mode(model_a, which)
    try:
        yield
    finally:
        for i, s in slots(model_a).items():
            s.mode = previous[i]


def trainable_parameters(model_a: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for slot in slots(model_a).values():
        params.extend(slot.trainable_parameters())
    return params


def freeze_all_but_projections(model_a: nn.Module, router_trainable: bool = False) -> None:
    """Freeze A and B entirely; W_in and W_out train. AB5 also unfreezes B's routers."""
    model_a.requires_grad_(False)
    for slot in slots(model_a).values():
        slot.w_in.requires_grad_(True)
        slot.w_out.requires_grad_(True)
        if router_trainable and hasattr(slot.inner, "make_router_trainable"):
            slot.inner.make_router_trainable()


def projection_state(model_a: nn.Module) -> dict[str, torch.Tensor]:
    """Everything a checkpoint needs: projections (and routers, if trained), per layer."""
    out: dict[str, torch.Tensor] = {}
    for i, slot in slots(model_a).items():
        for name, p in slot.named_parameters():
            if name.startswith(("w_in.", "w_out.")) or p.requires_grad:
                out[f"{i}.{name}"] = p.detach().cpu().clone()
    return out


def load_projection_state(model_a: nn.Module, state: dict[str, torch.Tensor], strict: bool = True) -> None:
    current = slots(model_a)
    seen = set()
    for key, tensor in state.items():
        i, name = key.split(".", 1)
        slot = current.get(int(i))
        if slot is None:
            if strict:
                raise KeyError(f"checkpoint has layer {i}, model has no transplant there")
            continue
        param = dict(slot.named_parameters())[name]
        with torch.no_grad():
            param.copy_(tensor.to(param.device, param.dtype))
        seen.add(key)
    if strict:
        expected = {f"{i}.{n}" for i, s in current.items() for n, _ in s.named_parameters() if n.startswith(("w_in.", "w_out."))}
        missing = expected - seen
        if missing:
            raise KeyError(f"checkpoint misses {sorted(missing)[:4]} ...")


@contextlib.contextmanager
def record_routing(model_a: nn.Module, on: bool = True):
    inners = [s.inner for s in slots(model_a).values() if hasattr(s.inner, "record_routing")]
    for inner in inners:
        inner.record_routing = on
        inner.router_calls = []
    try:
        yield
    finally:
        for inner in inners:
            inner.record_routing = False


def train_mode(model_a: nn.Module) -> None:
    """Training mode for Gemma's layers (activates gradient checkpointing) while the frozen
    MoE blocks and Gemma's own FFNs stay in eval mode. MobileMoE may switch to its
    pre-training drop-and-pad dispatch in train mode; the transplant must always use the
    dropless inference path (plan assumption A4)."""
    model_a.train()
    for slot in slots(model_a).values():
        slot.ffn_a.eval()
        slot.inner.eval()


def set_shared_expert(model_a: nn.Module, enabled: bool) -> None:
    for slot in slots(model_a).values():
        if hasattr(slot.inner, "shared_enabled"):
            slot.inner.shared_enabled = enabled


def first_slot(model_a: nn.Module) -> Optional[nn.Module]:
    s = slots(model_a)
    return s[min(s)] if s else None
