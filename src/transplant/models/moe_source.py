"""Find and extract MobileMoE's per-layer MoE blocks (pre-MoE RMSNorm + router + experts + shared expert).

MobileMoE ships custom code (``trust_remote_code=True``) whose module names were not
public when this was written (plan assumption A3). Every name below is therefore
either given in ``pair.b`` or discovered: run ``python -m transplant inspect`` at M0,
check what it found, and pin the names in ``configs/pairs/primary.yaml``.

The extraction equivalence test (``tests/test_real_models.py``) must pass before any
training: the extracted block, given B's own hidden states, must reproduce B's own
MoE-block output.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

LAYER_PATHS = ("model.layers", "model.decoder.layers", "transformer.h", "transformer.layers", "layers")
NORM_NAMES = (
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "pre_moe_layernorm",
    "ffn_norm",
    "post_attention_norm",
    "ln_2",
)
SHARED_NAMES = ("shared_expert", "shared_experts", "shared_mlp", "shared_ffn")


def resolve_attr(root: nn.Module, dotted: str) -> nn.Module:
    node = root
    for part in dotted.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _has_attr_path(root: nn.Module, dotted: str) -> bool:
    try:
        resolve_attr(root, dotted)
        return True
    except (AttributeError, IndexError, KeyError):
        return False


def find_layers(model: nn.Module, path: str = "auto", n_layers: Optional[int] = None) -> tuple[str, nn.ModuleList]:
    candidates = LAYER_PATHS if path == "auto" else (path,)
    for cand in candidates:
        if _has_attr_path(model, cand):
            layers = resolve_attr(model, cand)
            if isinstance(layers, nn.ModuleList) and (n_layers is None or len(layers) == n_layers):
                return cand, layers
    raise ValueError(f"could not find decoder layers (tried {candidates}); set pair.b.layers_path")


def _expert_list(module: nn.Module, num_experts: int) -> Optional[nn.ModuleList]:
    for sub in module.modules():
        if isinstance(sub, nn.ModuleList) and len(sub) == num_experts:
            return sub
    return None


def _router_linear(module: nn.Module, num_experts: int) -> Optional[tuple[str, nn.Linear]]:
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear) and sub.out_features == num_experts:
            return name, sub
    return None


def find_moe_attr(layer: nn.Module, num_experts: int, name: str = "auto") -> str:
    if name != "auto":
        return name
    for child_name, child in layer.named_children():
        if _expert_list(child, num_experts) is not None or _router_linear(child, num_experts) is not None:
            return child_name
    raise ValueError(
        f"no child of the decoder layer holds {num_experts} experts; children are "
        f"{[n for n, _ in layer.named_children()]}. Set pair.b.moe_attr."
    )


def find_norm_attr(layer: nn.Module, name: str = "auto") -> str:
    if name != "auto":
        return name
    for cand in NORM_NAMES:
        if hasattr(layer, cand):
            return cand
    raise ValueError(
        f"no pre-MoE norm found (tried {NORM_NAMES}); children are "
        f"{[n for n, _ in layer.named_children()]}. Set pair.b.norm_attr."
    )


def find_shared_attr(moe: nn.Module, name: str = "auto") -> Optional[str]:
    if name in (None, "none"):
        return None
    if name != "auto":
        return name
    for cand in SHARED_NAMES:
        if hasattr(moe, cand):
            return cand
    return None


def find_routing_bias(moe: nn.Module, num_experts: int, router_name: Optional[str] = None) -> Optional[torch.Tensor]:
    """A per-expert bias added to scores before top-k (aux-loss-free balancing), if any.

    The router Linear's own bias is already inside its logits, so it is excluded.
    """
    skip = (router_name + ".") if router_name else None
    for name, tensor in list(moe.named_parameters()) + list(moe.named_buffers()):
        if skip and name.startswith(skip):
            continue
        if "bias" in name.lower() and tensor.ndim == 1 and tensor.numel() == num_experts:
            return tensor
    return None


# ---------------------------------------------------------------------------------------
# A forward pass for MobileMoE's MoE block that the Apple GPU can run without stalling.
#
# MobileMoE's own forward (MobileMoETextMoe.forward, read at M0) loops over its 60 experts
# and, for each, finds the tokens routed to it with ``nonzero``. On MPS that costs a GPU
# sync per expert, and every expert gets a different number of tokens on every call, so
# every call brings new tensor shapes; MPS compiles and caches kernels per shape and the
# cache only grows (the first Stage 1 attempt reached a 29 GB footprint in ridge init).
# ``bucketed_moe_forward`` does the same arithmetic with one sync per layer and with each
# expert's token group padded to one of a few sizes (16, 24, 32, 48, 64, 96, ...).
# ---------------------------------------------------------------------------------------


def _bucket(n: int) -> int:
    """Smallest size >= n in {16, 24, 32, 48, 64, 96, 128, ...}: two sizes per octave, so a
    group is padded by less than half (about a fifth on average) and only about 2 * log2(n)
    shapes ever occur."""
    if n <= 16:
        return 16
    p = 1 << (n - 1).bit_length()
    q = (p >> 1) + (p >> 2)
    return q if n <= q else p


def has_mobilemoe_layout(moe: nn.Module) -> bool:
    """MobileMoETextMoe's parameter layout: experts.gate_up_proj (E, d, 2F), experts.down_proj
    (E, F, d), a router Linear, a shared expert and top_k."""
    experts = getattr(moe, "experts", None)
    gate_up, down = getattr(experts, "gate_up_proj", None), getattr(experts, "down_proj", None)
    return (
        torch.is_tensor(gate_up) and torch.is_tensor(down) and gate_up.ndim == 3 and down.ndim == 3
        and isinstance(getattr(moe, "router", None), nn.Linear)
        and hasattr(moe, "shared_expert") and hasattr(moe, "top_k") and hasattr(experts, "act_fn")
    )


def _route(self, x: torch.Tensor, k: int):
    """MobileMoE's routing: fp32 logits, sigmoid, top-k on scores + selection bias, the chosen
    unbiased scores normalized over the k and scaled. Returns (logits, (N, k) ids, (N, k) weights)."""
    router_logits = F.linear(x.float(), self.router.weight.float())
    scores = torch.sigmoid(router_logits)
    bias = getattr(self, "expert_bias", None)
    _, topk_idx = torch.topk(scores + bias if bias is not None else scores, k, dim=-1)
    topk_scores = scores.gather(1, topk_idx)
    if getattr(self, "norm_topk_prob", True):
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
    topk_scores = topk_scores * float(getattr(self, "routed_scaling_factor", 1.0))
    return router_logits, topk_idx, topk_scores


def bucketed_moe_forward(self, hidden_states: torch.Tensor):
    """MobileMoETextMoe.forward, same arithmetic: fp32 router logits, sigmoid, top-k on scores
    plus the selection bias, unbiased scores normalized over the k and scaled, experts
    weighted after their output, plus the shared expert. Returns (output (N, d), router logits),
    like the original. Routed outputs are summed in fp32 (the original sums in bf16)."""
    gate_up, down = self.experts.gate_up_proj, self.experts.down_proj
    num_experts, d = gate_up.shape[0], gate_up.shape[1]
    k = int(self.top_k)
    x = hidden_states.reshape(-1, d)
    n, dev = x.shape[0], x.device

    router_logits, topk_idx, topk_scores = _route(self, x, k)

    # Group the n * k (token, expert) assignments by expert, each group padded to a bucket size.
    flat_e = topk_idx.reshape(-1)
    counts = torch.bincount(flat_e, minlength=num_experts).tolist()  # the layer's only host sync
    sizes = [_bucket(c) if c else 0 for c in counts]
    pad_start, act_start, p_off, a_off = [], [], 0, 0
    for c, b in zip(counts, sizes):
        pad_start.append(p_off)
        act_start.append(a_off)
        p_off += b
        a_off += c
    total = _bucket(p_off)
    sorted_e, order = torch.sort(flat_e.to(torch.int32), stable=True)
    sorted_e = sorted_e.long()
    starts = torch.tensor([pad_start, act_start], dtype=torch.long, device=dev)
    dest = starts[0][sorted_e] + torch.arange(n * k, device=dev) - starts[1][sorted_e]
    table = torch.full((total,), n, dtype=torch.long, device=dev)  # row n of x_ext is zeros
    table[dest] = torch.div(order, k, rounding_mode="floor")
    weights = torch.zeros(total, dtype=torch.float32, device=dev)
    weights[dest] = topk_scores.reshape(-1)[order]

    x_ext = torch.cat([x, x.new_zeros(1, d)])
    out = torch.zeros(n + 1, d, dtype=torch.float32, device=dev)
    act = self.experts.act_fn
    for e, (c, b) in enumerate(zip(counts, sizes)):
        if c == 0:
            continue
        s = pad_start[e]
        idx = table[s:s + b]
        gate, up = (x_ext[idx] @ gate_up[e]).chunk(2, dim=-1)
        y = (act(gate) * up) @ down[e]
        out.index_add_(0, idx, y.float() * weights[s:s + b, None])
    out = out[:n] + self.shared_expert(x).float()
    return out.to(x.dtype), router_logits


def grouped_moe_forward(self, hidden_states: torch.Tensor):
    """The same arithmetic as ``bucketed_moe_forward``, with all experts in one batched matrix
    product: every expert's token group is padded to the largest group (rounded up to a bucket
    size), so three ``bmm`` calls replace about 400 small kernels per layer. The price is the
    padding: when routing is uneven most rows of the batch are zeros. Which of the two is
    faster on a given GPU is measured, not assumed: ``transplant memcheck`` prints both phases'
    speed for the forward set in ``pair.b.moe_forward``."""
    gate_up, down = self.experts.gate_up_proj, self.experts.down_proj
    num_experts, d = gate_up.shape[0], gate_up.shape[1]
    k = int(self.top_k)
    x = hidden_states.reshape(-1, d)
    n, dev = x.shape[0], x.device
    router_logits, topk_idx, topk_scores = _route(self, x, k)

    flat_e = topk_idx.reshape(-1)
    counts = torch.bincount(flat_e, minlength=num_experts)
    cap = _bucket(int(counts.max()))  # the layer's only host sync
    sorted_e, order = torch.sort(flat_e.to(torch.int32), stable=True)
    sorted_e = sorted_e.long()
    starts = torch.cumsum(counts, 0) - counts
    slot = sorted_e * cap + torch.arange(n * k, device=dev) - starts[sorted_e]
    table = torch.full((num_experts * cap,), n, dtype=torch.long, device=dev)  # row n of x_ext is zeros
    table[slot] = torch.div(order, k, rounding_mode="floor")
    weights = torch.zeros(num_experts * cap, dtype=torch.float32, device=dev)
    weights[slot] = topk_scores.reshape(-1)[order]

    x_ext = torch.cat([x, x.new_zeros(1, d)])
    gate, up = torch.bmm(x_ext[table].view(num_experts, cap, d), gate_up).chunk(2, dim=-1)
    y = torch.bmm(self.experts.act_fn(gate) * up, down).reshape(num_experts * cap, d)
    out = torch.zeros(n + 1, d, dtype=torch.float32, device=dev)
    out.index_add_(0, table, y.float() * weights[:, None])
    out = out[:n] + self.shared_expert(x).float()
    return out.to(x.dtype), router_logits


FAST_FORWARDS = {"bucketed": bucketed_moe_forward, "grouped": grouped_moe_forward}
_SWAPPED: dict[tuple[type, str], type] = {}


def use_fast_forward(model_b: nn.Module, b_cfg: dict, kind: str = "bucketed") -> int:
    """Give every MobileMoE-style MoE block of ``model_b`` the ``kind`` forward (in B's own
    forward pass and in the extracted blocks alike). Returns how many blocks changed.
    The original stays callable as ``native_forward``."""
    if kind not in FAST_FORWARDS:
        raise ValueError(f"pair.b.moe_forward must be native or one of {sorted(FAST_FORWARDS)}")
    spec = MoESpec.from_cfg(b_cfg)
    _, layers = find_layers(model_b, spec.layers_path)
    changed = 0
    for layer in layers:
        moe = getattr(layer, find_moe_attr(layer, spec.num_experts, spec.moe_attr))
        cls = type(moe)
        if getattr(cls, "fast_forward_kind", None) == kind or not has_mobilemoe_layout(moe):
            continue
        base = getattr(cls, "native_class", cls)
        if (base, kind) not in _SWAPPED:
            _SWAPPED[(base, kind)] = type(f"{kind.capitalize()}{base.__name__}", (base,),
                                          {"forward": FAST_FORWARDS[kind], "native_forward": base.forward,
                                           "native_class": base, "fast_forward_kind": kind})
        moe.__class__ = _SWAPPED[(base, kind)]
        changed += 1
    return changed


def use_bucketed_forward(model_b: nn.Module, b_cfg: dict) -> int:
    return use_fast_forward(model_b, b_cfg, "bucketed")


@dataclass
class MoESpec:
    num_experts: int = 60
    top_k: int = 4
    layers_path: str = "auto"
    moe_attr: str = "auto"
    norm_attr: str = "auto"
    router_attr: str = "auto"
    shared_attr: str = "auto"

    @classmethod
    def from_cfg(cls, b_cfg: dict) -> "MoESpec":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: b_cfg[k] for k in keys if k in b_cfg})


class MoESourceBlock(nn.Module):
    """B's frozen pre-MoE RMSNorm N_B followed by its MoE block MoE_B.

    forward(u) = MoE_B(N_B(u)). Router logits of the last call are kept in
    ``last_router_logits`` while ``record_routing`` is True (routing diagnostics).
    """

    def __init__(
        self,
        norm: nn.Module,
        moe: nn.Module,
        *,
        layer_index: int,
        spec: MoESpec,
        router_name: Optional[str],
        shared_name: Optional[str],
    ):
        super().__init__()
        self.norm = norm
        self.moe = moe
        self.layer_index = layer_index
        self.num_experts = spec.num_experts
        self.top_k = spec.top_k
        self.router_name = router_name
        self.shared_name = shared_name
        self.record_routing = False
        # One entry per forward call while record_routing is on: (N_tokens, num_experts) logits.
        self.router_calls: list[torch.Tensor] = []
        self.shared_enabled = True
        self.spec = spec
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        if self._hooks:
            return
        # Router logits normally come from the router module's own forward. MobileMoE skips
        # router.forward (it calls F.linear on router.weight in fp32) and returns the logits as
        # the MoE block's second output, so the block's output is read as well.
        self._hooks.append(self.moe.register_forward_pre_hook(self._moe_pre_hook))
        self._hooks.append(self.moe.register_forward_hook(self._moe_hook))
        if self.router_name is not None:
            router = resolve_attr(self.moe, self.router_name) if self.router_name else self.moe
            self._hooks.append(router.register_forward_hook(self._router_hook))
        if self.shared_name is not None:
            shared = resolve_attr(self.moe, self.shared_name)
            self._hooks.append(shared.register_forward_hook(self._shared_hook))

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    # -- hooks -------------------------------------------------------------
    def _moe_pre_hook(self, module, inputs):
        self._router_seen = False

    def _router_hook(self, module, inputs, output):
        self._router_seen = True
        if self.record_routing:
            logits = output[0] if isinstance(output, tuple) else output
            self.router_calls.append(logits.detach().reshape(-1, logits.shape[-1]))

    def _moe_hook(self, module, inputs, output):
        if not self.record_routing or getattr(self, "_router_seen", False):
            return None
        if isinstance(output, tuple):  # (hidden, router_logits, ...): take the (N, num_experts) logits
            for item in output[1:]:
                if torch.is_tensor(item) and item.shape[-1] == self.num_experts:
                    self.router_calls.append(item.detach().reshape(-1, self.num_experts))
                    break
        return None

    def _shared_hook(self, module, inputs, output):
        if not self.shared_enabled:  # ablation AB6: routed experts only
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + tuple(output[1:])
            return torch.zeros_like(output)
        return None

    # -- api ---------------------------------------------------------------
    @property
    def router(self) -> Optional[nn.Linear]:
        if self.router_name is None:
            return None
        return resolve_attr(self.moe, self.router_name)

    def routing_bias(self) -> Optional[torch.Tensor]:
        return find_routing_bias(self.moe, self.num_experts, self.router_name)

    def make_router_trainable(self) -> None:
        """Ablation AB5. The router trains in fp32 (fp16 weights cannot take loss-scaled
        AdamW updates): its input is cast to fp32 and its logits back to the block dtype."""
        router = self.router
        if router is None:
            raise ValueError("no router found in this MoE block; set pair.b.router_attr")
        if getattr(router, "_fp32_trainable", False):
            return
        router.float()
        router.requires_grad_(True)

        def pre(module, inputs):
            self._router_in_dtype = inputs[0].dtype
            return (inputs[0].float(),) + tuple(inputs[1:])

        def post(module, inputs, output):
            dt = getattr(self, "_router_in_dtype", torch.float32)
            if isinstance(output, tuple):
                return (output[0].to(dt),) + tuple(output[1:])
            return output.to(dt)

        router.register_forward_pre_hook(pre)
        router.register_forward_hook(post)
        router._fp32_trainable = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        out = self.moe(self.norm(u))
        if isinstance(out, tuple):
            out = out[0]
        elif hasattr(out, "hidden_states") and not torch.is_tensor(out):
            out = out.hidden_states
        # MobileMoE returns its output flattened to (batch * seq, d); give it u's shape back.
        return out.reshape(u.shape)

    def selected_experts(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-k expert ids per token for recorded router logits: (N, k).

        Sigmoid scores, plus the per-expert selection bias when B has one (checked at M0).
        """
        scores = torch.sigmoid(logits.float())
        bias = self.routing_bias()
        if bias is not None:
            scores = scores + bias.detach().float().to(scores.device)
        return scores.topk(self.top_k, dim=-1).indices


def extract_blocks(model_b: nn.Module, b_cfg: dict, b_layers: list[int]) -> dict[int, MoESourceBlock]:
    """Wrap B's norm + MoE modules for the requested layers. Shares (does not copy) weights."""
    spec = MoESpec.from_cfg(b_cfg)
    _, layers = find_layers(model_b, spec.layers_path)
    blocks: dict[int, MoESourceBlock] = {}
    for j in sorted(set(b_layers)):
        layer = layers[j]
        moe_name = find_moe_attr(layer, spec.num_experts, spec.moe_attr)
        norm_name = find_norm_attr(layer, spec.norm_attr)
        moe = getattr(layer, moe_name)
        norm = getattr(layer, norm_name)
        if spec.router_attr == "auto":
            found = _router_linear(moe, spec.num_experts)
            router_name = found[0] if found else None
        else:
            router_name = spec.router_attr
        shared_name = find_shared_attr(moe, spec.shared_attr)
        blocks[j] = MoESourceBlock(
            norm, moe, layer_index=j, spec=spec, router_name=router_name, shared_name=shared_name
        )
    return blocks


def describe_layer(model_b: nn.Module, b_cfg: dict, j: int = 0) -> dict:
    """What discovery finds in one B layer (for ``transplant inspect``)."""
    spec = MoESpec.from_cfg(b_cfg)
    path, layers = find_layers(model_b, spec.layers_path)
    layer = layers[j]
    moe_name = find_moe_attr(layer, spec.num_experts, spec.moe_attr)
    moe = getattr(layer, moe_name)
    router = _router_linear(moe, spec.num_experts)
    bias = find_routing_bias(moe, spec.num_experts, router[0] if router else None)
    return {
        "layers_path": path,
        "n_layers": len(layers),
        "layer_children": [n for n, _ in layer.named_children()],
        "moe_attr": moe_name,
        "norm_attr": find_norm_attr(layer, spec.norm_attr),
        "router_attr": router[0] if router else None,
        "shared_attr": find_shared_attr(moe, spec.shared_attr),
        "routing_bias_found": bias is not None,
        "moe_params": sum(p.numel() for p in moe.parameters()),
        "moe_class": type(moe).__name__,
    }


@torch.no_grad()
def randomize_like(block: MoESourceBlock, seed: int = 0) -> MoESourceBlock:
    """Control C5: same MoE architecture, weights redrawn (per tensor, matching mean and std).

    Norm gains are kept so the input scale matches; every other weight is random. Frozen.
    """
    block._remove_hooks()  # hooks hold bound methods; copy the modules without them
    try:
        norm, moe = copy.deepcopy(block.norm), copy.deepcopy(block.moe)
    finally:
        block._install_hooks()
    clone = MoESourceBlock(
        norm,
        moe,
        layer_index=block.layer_index,
        spec=block.spec,
        router_name=block.router_name,
        shared_name=block.shared_name,
    )
    gen = torch.Generator().manual_seed(seed + 1000 * block.layer_index)
    for name, param in clone.moe.named_parameters():
        if param.ndim < 2:
            continue
        mean, std = param.float().mean().item(), param.float().std().item()
        fresh = torch.randn(param.shape, generator=gen) * std + mean
        param.copy_(fresh.to(param.dtype))
    clone.requires_grad_(False)
    return clone
