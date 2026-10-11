"""The bucketed MoE forward (used on the Apple GPU) computes exactly what MobileMoE's own forward does."""
import math

import pytest
import torch

from transplant.models.load import load_model_b_full
from transplant.models.moe_source import _bucket, bucketed_moe_forward, grouped_moe_forward, use_bucketed_forward, use_fast_forward
from transplant.models.tiny import TINY_B_CFG, TinySparseMoE, tiny_model_b


def _moe(seed=0, d=24, experts=12, k=4, bias=True):
    torch.manual_seed(seed)
    moe = TinySparseMoE(d, num_experts=experts, top_k=k, expert_hidden=16, shared_hidden=32)
    if bias:
        with torch.no_grad():
            moe.expert_bias.copy_(torch.randn(experts) * 0.3)
    return moe


@pytest.mark.parametrize("forward", [bucketed_moe_forward, grouped_moe_forward])
@pytest.mark.parametrize("n", [1, 3, 37, 500])
def test_fast_forwards_equal_native_values_logits_and_grads(n, forward):
    moe = _moe()
    x = torch.randn(n, 24, requires_grad=True)
    ref, ref_logits = moe(x)  # TinySparseMoE.forward is MobileMoE's forward, line for line
    ref.pow(2).sum().backward()
    g_ref = x.grad.clone()
    x.grad = None
    out, logits = forward(moe, x)
    out.pow(2).sum().backward()
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, g_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("forward", [bucketed_moe_forward, grouped_moe_forward])
def test_fast_forwards_handle_batched_input_and_bf16(forward):
    moe = _moe(seed=1).to(torch.bfloat16)
    x = torch.randn(2, 9, 24, dtype=torch.bfloat16)
    ref, _ = moe(x)
    out, _ = forward(moe, x)
    assert out.shape == (18, 24) and out.dtype == torch.bfloat16  # flattened, like MobileMoE
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_bucket_sizes_are_few_and_never_too_small():
    sizes = {_bucket(n) for n in range(1, 20_000)}
    assert all(_bucket(n) >= n for n in range(1, 20_000))
    assert all(_bucket(n) < 1.51 * n for n in range(17, 20_000))
    assert len(sizes) <= 2 * math.log2(20_000)


def test_loading_b_switches_its_moe_blocks_to_the_bucketed_forward():
    model = load_model_b_full({"model_id": "tiny", **TINY_B_CFG}, torch.float32)
    moes = [layer.mlp for layer in model.model.layers]
    assert all(type(m).__name__ == "BucketedTinySparseMoE" for m in moes)
    assert use_bucketed_forward(model, TINY_B_CFG) == 0  # already switched: nothing to do
    native = tiny_model_b()
    ids = torch.randint(0, 50, (2, 7))
    torch.testing.assert_close(model.model(ids), native.model(ids), rtol=1e-5, atol=1e-5)


def test_switching_between_fast_forwards_keeps_the_native_one():
    model = tiny_model_b()
    assert use_fast_forward(model, TINY_B_CFG, "grouped") == 4
    moe = model.model.layers[0].mlp
    assert type(moe).__name__ == "GroupedTinySparseMoE"
    assert use_fast_forward(model, TINY_B_CFG, "bucketed") == 4
    assert type(moe).__name__ == "BucketedTinySparseMoE" and moe.native_class is TinySparseMoE
    x = torch.randn(30, 48)
    torch.testing.assert_close(moe(x)[0], moe.native_forward(x)[0], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        use_fast_forward(model, TINY_B_CFG, "fastest")
