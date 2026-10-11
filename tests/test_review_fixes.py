"""Regression tests for issues found in code review (dtype, alignment, seeds, resume, checkpoints)."""
import time

import torch

from transplant.data.sources import DocSource, SourceSpec
from transplant.models import surgery
from transplant.models.align import aligned_pairs
from transplant.models.load import build_hybrid, cast_keep_rope_fp32
from transplant.models.moe_source import find_routing_bias
from transplant.models.tiny import tiny_model_a
from transplant.train.checkpoint import ckpt_due
from transplant.utils.device import PrecisionPolicy


def test_rope_buffers_stay_fp32_when_model_is_cast_to_bf16():
    model = tiny_model_a()
    cast_keep_rope_fp32(model, torch.bfloat16, torch.device("cpu"))
    bufs = {n: b for n, b in model.named_buffers() if "inv_freq" in n}
    assert bufs and all(b.dtype == torch.float32 for b in bufs.values())
    assert model.model.layers[0].mlp.gate_proj.weight.dtype == torch.bfloat16


def test_alignment_keeps_one_token_per_end_offset():
    # A character split into three byte tokens on the B side: all end at offset 5.
    a = [(0, 0), (0, 5), (5, 9)]
    b = [(0, 0), (0, 4), (4, 5), (4, 5), (4, 5), (5, 9)]
    assert aligned_pairs(a, b) == [(1, 4), (2, 5)]


def test_router_trains_in_fp32_when_blocks_are_fp16(tiny_cfg, cpu):
    tiny_cfg.transplant.router_trainable = True
    policy = PrecisionPolicy(torch.float32, torch.float16, torch.float32, False)
    hybrid = build_hybrid(tiny_cfg, cpu, policy)
    slot = surgery.first_slot(hybrid.model)
    router = slot.inner.router
    assert router.weight.dtype == torch.float32 and router.weight.requires_grad
    assert slot.inner.moe.experts.gate_up_proj.dtype == torch.float16
    out = slot.transplant(torch.randn(1, 12, hybrid.model.config.hidden_size))
    out.float().pow(2).mean().backward()
    assert router.weight.grad is not None and router.weight.grad.dtype == torch.float32


def test_routing_bias_ignores_the_routers_own_bias():
    moe = torch.nn.Module()
    moe.gate = torch.nn.Linear(8, 6, bias=True)
    assert find_routing_bias(moe, 6, "gate") is None
    moe.register_buffer("e_score_correction_bias", torch.zeros(6))
    assert find_routing_bias(moe, 6, "gate") is moe.e_score_correction_bias


def test_seeds_read_different_documents():
    spec = SourceSpec(name="web", weight=1.0, kind="synthetic", seed=4)
    take = lambda seed: [next(it) for it in [DocSource(spec, subsample_seed=seed).open(0)] for _ in range(40)]  # noqa: E731
    s0, s0b, s1 = take(0), take(0), take(1)
    assert s0 == s0b and s0 != s1
    assert len(set(s0) & set(s1)) > 20  # mostly overlapping data, different 1-in-8 left out


def test_resumed_source_keeps_its_state_until_read():
    from datasets import Dataset

    class Local(DocSource):
        def _base_dataset(self):
            return Dataset.from_dict({"text": [f"doc {i}" for i in range(50)]}).to_iterable_dataset(num_shards=2)

    spec = SourceSpec(name="x", weight=1.0)
    src = Local(spec)
    it = src.open(3)
    for _ in range(7):
        next(it)
    state = src.state_dict()
    again = Local(spec)
    again.open(3, state)
    assert again.state_dict() == state  # nothing read yet: still the loaded position


def test_checkpoint_due_by_steps_or_minutes():
    cfg = {"checkpoint": {"every_steps": 200, "every_minutes": 30}}
    now = time.time()
    assert ckpt_due(cfg, 199, now)
    assert not ckpt_due(cfg, 10, now)
    assert ckpt_due(cfg, 10, now - 31 * 60)
