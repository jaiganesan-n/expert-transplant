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
