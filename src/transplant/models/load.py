"""Load Gemma (A), MobileMoE (B) and their tokenizers; build the Replace hybrid.

B's attention, embeddings and output head never reach the GPU during training: B is
loaded on the CPU, its norm + MoE modules for the needed layers are kept, and the rest
is dropped before anything moves to the accelerator.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from . import surgery
from .layer_map import resolve_layer_map
from .moe_source import MoESourceBlock, extract_blocks, randomize_like
from .projection import load_projection_matrix, make_projection
from ..utils.device import PrecisionPolicy

INNER_KINDS = ("moe_b", "random_moe_b", "a_ffn_bottleneck")


def cast_keep_rope_fp32(model, dtype: torch.dtype, device) -> None:
    """Move/cast a model without downcasting its rotary-embedding frequency buffers.

    ``from_pretrained(torch_dtype=...)`` keeps ``inv_freq`` in fp32; a blanket
    ``model.to(dtype)`` would cast it to bf16/fp16 and corrupt positions far into the
    sequence. Parameters and other buffers get ``dtype``; any ``*inv_freq`` buffer stays fp32.
    """
    model.to(device=device)
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is None or not buf.is_floating_point():
                continue
            module._buffers[name] = buf.to(torch.float32 if "inv_freq" in name else dtype)
        for name, param in module._parameters.items():
            if param is not None and param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(dtype)


def is_tiny(side_cfg: dict) -> bool:
    return str(side_cfg.get("model_id", "")).startswith("tiny")


def load_tokenizer(side_cfg: dict, side: str):
    if is_tiny(side_cfg):
        from .tiny import tiny_tokenizer

        return tiny_tokenizer(side)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        side_cfg["model_id"],
        revision=side_cfg.get("revision"),
        trust_remote_code=bool(side_cfg.get("trust_remote_code", False)),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_a(a_cfg: dict, dtype: torch.dtype, device: torch.device, attn_implementation: str = "sdpa"):
    if is_tiny(a_cfg):
        from .tiny import tiny_model_a

        model = tiny_model_a(seed=int(a_cfg.get("seed", 0)), pretrain_steps=int(a_cfg.get("pretrain_steps", 0)))
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            a_cfg["model_id"],
            revision=a_cfg.get("revision"),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    cast_keep_rope_fp32(model, dtype, device)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    return model


def load_model_b_full(b_cfg: dict, dtype: torch.dtype, device: torch.device | str = "cpu"):
    """The whole of B: needed only for calibration (ridge W_in), bits-per-byte and routing agreement."""
    if is_tiny(b_cfg):
        from .tiny import tiny_model_b

        model = tiny_model_b(seed=int(b_cfg.get("seed", 1)), pretrain_steps=int(b_cfg.get("pretrain_steps", 0)),
                             n_layers=int(b_cfg.get("num_layers") or 4))
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            b_cfg["model_id"],
            revision=b_cfg.get("revision"),
            torch_dtype=dtype,
            trust_remote_code=bool(b_cfg.get("trust_remote_code", True)),
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
    kind = b_cfg.get("moe_forward", "bucketed")
    if kind != "native":
        from .moe_source import use_fast_forward

        use_fast_forward(model, b_cfg, kind)  # MobileMoE's own per-expert loop stalls the Apple GPU
    cast_keep_rope_fp32(model, dtype, device)
    model.requires_grad_(False)
    model.eval()
    return model


def n_layers_of(model) -> int:
    return int(model.config.num_hidden_layers)


def hidden_of(model) -> int:
    return int(model.config.hidden_size)


def b_architecture(b_cfg: dict) -> tuple[int, int]:
    """(L_B, d_B) from config, so the B weights need not be loaded to plan the surgery."""
    if is_tiny(b_cfg):
        from .tiny import tiny_model_b

        m = tiny_model_b(n_layers=int(b_cfg.get("num_layers") or 4))
        return n_layers_of(m), hidden_of(m)
    return int(b_cfg["num_layers"]), int(b_cfg["hidden_size"])


@dataclass
class Hybrid:
    model: nn.Module  # Gemma with ReplaceFFN slots attached
    tokenizer: object
    layer_map: dict[int, int]  # A layer -> B layer
    blocks: dict[int, nn.Module] = field(default_factory=dict)  # B layer -> MoESourceBlock
    policy: Optional[PrecisionPolicy] = None
    inner_kind: str = "moe_b"

    @property
    def a_layers(self) -> list[int]:
        return sorted(self.layer_map)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return surgery.trainable_parameters(self.model)


def load_moe_blocks(
    b_cfg: dict, b_layers: list[int], policy: PrecisionPolicy, device: torch.device, randomize: bool = False, seed: int = 0
) -> dict[int, MoESourceBlock]:
    model_b = load_model_b_full(b_cfg, dtype=policy.b_dtype, device="cpu")
    blocks = extract_blocks(model_b, b_cfg, b_layers)
    del model_b
    gc.collect()
    out = {}
    for j, block in blocks.items():
        if randomize:
            block = randomize_like(block, seed=seed)
        block.to(device=device, dtype=policy.b_dtype)
        block.requires_grad_(False)
        block.eval()
        out[j] = block
    return out


def build_hybrid(
    cfg: dict,
    device: torch.device,
    policy: PrecisionPolicy,
    *,
    init_matrices: Optional[dict] = None,
    model_a: Optional[nn.Module] = None,
    blocks: Optional[dict[int, nn.Module]] = None,
) -> Hybrid:
    """Gemma with B's MoE blocks in the FFN slots of ``pair.layers``.

    ``init_matrices``: {A layer: {"w_in": (d_B, d_A), "w_out": (d_A, d_B)}} from ridge
    init. Without it, projections keep PyTorch's default random init (ablation AB1).
    """
    pair, tcfg = cfg["pair"], cfg["transplant"]
    inner_kind = tcfg.get("inner", "moe_b")
    if inner_kind not in INNER_KINDS:
        raise ValueError(f"transplant.inner must be one of {INNER_KINDS}")
    tok_a = load_tokenizer(pair["a"], "a")
    if model_a is None:
        model_a = load_model_a(pair["a"], policy.a_dtype, device, cfg.get("precision", {}).get("attn_implementation", "sdpa"))
    n_a, d_a = n_layers_of(model_a), hidden_of(model_a)
    n_b, d_b = b_architecture(pair["b"])
    a_layers, layer_map = resolve_layer_map(pair, n_a, n_b)

    if inner_kind == "a_ffn_bottleneck":
        blocks = {}
    elif blocks is None:
        blocks = load_moe_blocks(
            pair["b"],
            sorted(set(layer_map.values())),
            policy,
            device,
            randomize=(inner_kind == "random_moe_b"),
            seed=int(cfg.get("run", {}).get("seed", 0)),
        )

    kind = tcfg.get("projection", "linear")
    layers = surgery.decoder_layers(model_a)
    new_slots: dict[int, nn.Module] = {}
    for i, j in layer_map.items():
        ffn_a = layers[i].mlp
        if isinstance(ffn_a, surgery.SLOT_TYPES):
            raise RuntimeError("model already has transplants attached")
        w_in = make_projection(kind, d_a, d_b, rank=tcfg.get("rank", 256), hidden=tcfg.get("mlp_hidden", 2048))
        w_out = make_projection(kind, d_b, d_a, rank=tcfg.get("rank", 256), hidden=tcfg.get("mlp_hidden", 2048))
        if init_matrices is not None and i in init_matrices:
            load_projection_matrix(w_in, init_matrices[i]["w_in"])
            load_projection_matrix(w_out, init_matrices[i]["w_out"])
        w_in.to(device=device, dtype=policy.proj_dtype)
        w_out.to(device=device, dtype=policy.proj_dtype)
        if inner_kind == "a_ffn_bottleneck":
            slot = surgery.BottleneckFFN(ffn_a, w_in, w_out, layer_a=i, proj_dtype=policy.proj_dtype)
        else:
            slot = surgery.ReplaceFFN(
                ffn_a,
                blocks[j],
                w_in,
                w_out,
                layer_a=i,
                layer_b=j,
                b_dtype=policy.b_dtype,
                proj_dtype=policy.proj_dtype,
            )
        new_slots[i] = slot
    surgery.attach(model_a, new_slots)
    surgery.freeze_all_but_projections(model_a, router_trainable=bool(tcfg.get("router_trainable", False)))
    surgery.set_shared_expert(model_a, bool(tcfg.get("shared_expert", True)))
    return Hybrid(model_a, tok_a, layer_map, blocks, policy, inner_kind)
