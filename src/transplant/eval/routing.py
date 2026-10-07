"""Routing diagnostics for each transplanted layer.

* Health: normalized entropy of expert load (entropy / log E; alarm below 0.7),
  max-load fraction (share of assignments to the busiest expert), dead experts
  (under 0.1% of assignments).
* Agreement with B's native routing: mean Jaccard overlap of the hybrid's top-k
  experts and B's own top-k, on positions where both tokenizers end a token at the
  same character. Chance for two random 4-of-60 sets is about 0.039.
* Representation match: linear CKA between W_in N_A(h) and B's native MoE input
  (both RMS-normalized per token).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import torch

from ..models import surgery
from ..models.align import encode_aligned
from ..models.capture import capture_module_inputs, run_until
from .cka import linear_cka


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def chance_jaccard(num_experts: int = 60, k: int = 4) -> float:
    """Expected Jaccard index of two independent uniformly random k-subsets of E experts."""
    total = math.comb(num_experts, k)
    exp = 0.0
    for overlap in range(0, k + 1):
        p = math.comb(k, overlap) * math.comb(num_experts - k, k - overlap) / total
        exp += p * overlap / (2 * k - overlap)
    return exp


def routing_health(selected: torch.Tensor, num_experts: int, dead_threshold: float = 1e-3) -> dict:
    counts = torch.bincount(selected.reshape(-1).cpu(), minlength=num_experts).double()
    share = counts / counts.sum().clamp_min(1)
    nz = share[share > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {
        "normalized_entropy": entropy / math.log(num_experts),
        "max_load_fraction": float(share.max()),
        "dead_experts": int((share < dead_threshold).sum()),
        "tokens": int(selected.shape[0]),
    }


class RoutingCollector:
    """Collects top-k selections per transplanted layer across forward passes."""

    def __init__(self, model):
        self.model = model
        self.selected: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.slots = surgery.slots(model)
        by_block: dict[int, list[int]] = defaultdict(list)
        for i, slot in sorted(self.slots.items()):
            if hasattr(slot.inner, "router_calls"):
                by_block[id(slot.inner)].append(i)
        self.by_block = by_block
        self.blocks = {id(s.inner): s.inner for s in self.slots.values() if hasattr(s.inner, "router_calls")}

    def __call__(self, _ids=None) -> None:
        for key, layers in self.by_block.items():
            block = self.blocks[key]
            calls = block.router_calls
            # One call per layer that uses this block, in layer order.
            for layer, logits in zip(layers, calls):
                self.selected[layer].append(block.selected_experts(logits).cpu())
            block.router_calls = []

    def health(self) -> dict[int, dict]:
        out = {}
        for i, chunks in sorted(self.selected.items()):
            num_experts = self.slots[i].inner.num_experts
            out[i] = routing_health(torch.cat(chunks), num_experts)
        return out


@torch.no_grad()
def routing_agreement(hybrid, model_b_blocks: dict, model_b, tok_b, texts: list[str], seq_len: int, device, batch_size: int = 4) -> dict:
    """Jaccard of top-k sets and CKA of MoE inputs, hybrid vs B native, on aligned positions.

    ``model_b_blocks``: MoESourceBlocks extracted from the full ``model_b`` (B's own modules).
    """
    model = hybrid.model
    slots = surgery.slots(model)
    jacc = defaultdict(list)
    xs, ys = defaultdict(list), defaultdict(list)
    for k in range(0, len(texts), batch_size):
        ab = encode_aligned(hybrid.tokenizer, tok_b, texts[k : k + batch_size], seq_len)
        rows, pa, pb = ab.index_tensors(device)
        if rows.numel() == 0:
            continue
        # hybrid: record its routing and its W_in N_A(h)
        u_store = {}
        handles = []
        for i, slot in slots.items():
            def hook(mod, inp, out, i=i):
                u_store[i] = out.detach()

            handles.append(slot.w_in.register_forward_hook(hook))
        with surgery.mode(model, "transplant"), surgery.record_routing(model):
            model.model(input_ids=ab.ids_a.to(device), attention_mask=ab.mask_a.to(device), use_cache=False)
            collector = RoutingCollector(model)
            collector()
        for h in handles:
            h.remove()
        # B native: its own routing and its MoE-block inputs
        for blk in model_b_blocks.values():
            blk.record_routing = True
            blk.router_calls = []
        norms = {j: blk.norm for j, blk in model_b_blocks.items()}
        with capture_module_inputs(norms) as hb:
            run_until(model_b, input_ids=ab.ids_b.to(device), attention_mask=ab.mask_b.to(device))
        b_sel = {}
        for j, blk in model_b_blocks.items():
            if blk.router_calls:
                b_sel[j] = blk.selected_experts(blk.router_calls[0]).reshape(ab.ids_b.shape[0], ab.ids_b.shape[1], -1)
            blk.record_routing = False
            blk.router_calls = []
        for i, slot in slots.items():
            j = slot.layer_b
            if i not in collector.selected or j not in b_sel:
                continue
            h_sel = collector.selected[i][0].reshape(ab.ids_a.shape[0], ab.ids_a.shape[1], -1).to(device)
            a_sets = h_sel[rows, pa]
            b_sets = b_sel[j].to(device)[rows, pb]
            inter = (a_sets[:, :, None] == b_sets[:, None, :]).any(-1).sum(-1).float()
            kk = a_sets.shape[-1]
            jacc[i].append(inter / (2 * kk - inter))
            xs[i].append(u_store[i][rows, pa].float().cpu())
            ys[i].append(hb[j][rows, pb].float().cpu())
    out = {}
    for i in sorted(jacc):
        out[i] = {
            "jaccard": float(torch.cat(jacc[i]).mean()),
            # Per-token RMS-normalized on both sides: N_B discards scale, so compare directions.
            "cka_win_vs_b_input": linear_cka(_rms(torch.cat(xs[i])), _rms(torch.cat(ys[i]))),
            "pairs": int(sum(t.numel() for t in jacc[i])),
        }
    slot0 = next(iter(slots.values()))
    if hasattr(slot0.inner, "num_experts"):
        out["chance_jaccard"] = chance_jaccard(slot0.inner.num_experts, slot0.inner.top_k)
    return out
