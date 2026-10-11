"""Gating test 2 on the tiny B: the extracted N_B + MoE_B, given B's own hidden states,
reproduces B's own MoE-block output. Plus discovery and the random-weight control."""
import torch

from transplant.models.capture import capture_module_inputs
from transplant.models.moe_source import describe_layer, extract_blocks, randomize_like
from transplant.models.tiny import TINY_B_CFG, tiny_model_b


def test_extracted_block_reproduces_b():
    model = tiny_model_b(seed=1)
    layers = model.model.layers
    ids = torch.randint(4, model.config.vocab_size, (2, 20))
    outs, ins = {}, {}
    hooks = []
    for j, layer in enumerate(layers):
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, i, j=j: ins.__setitem__(j, i[0].detach())))
        hooks.append(layer.mlp.register_forward_hook(lambda m, i, o, j=j: outs.__setitem__(j, o[0].detach())))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    blocks = extract_blocks(model, TINY_B_CFG, list(range(len(layers))))
    for j, block in blocks.items():
        with torch.no_grad():
            got = block(ins[j])
        assert got.shape == ins[j].shape  # flattened MoE output gets the input's shape back
        torch.testing.assert_close(got.reshape(outs[j].shape), outs[j], rtol=1e-6, atol=1e-6)


def test_discovery_finds_module_names():
    info = describe_layer(tiny_model_b(), TINY_B_CFG, 0)
    assert info["moe_attr"] == "mlp"
    assert info["norm_attr"] == "post_attention_layernorm"
    assert info["router_attr"] == "router"
    assert info["shared_attr"] == "shared_expert"


def test_router_logits_recorded_per_call():
    model = tiny_model_b()
    blocks = extract_blocks(model, TINY_B_CFG, [0])
    block = blocks[0]
    block.record_routing = True
    u = torch.randn(3, 5, model.config.hidden_size)
    with torch.no_grad():
        block(u)
        block(u)
    assert len(block.router_calls) == 2
    sel = block.selected_experts(block.router_calls[0])
    assert sel.shape == (15, TINY_B_CFG["top_k"])


def test_random_moe_control_is_a_frozen_fresh_copy():
    model = tiny_model_b()
    block = extract_blocks(model, TINY_B_CFG, [1])[1]
    before = {n: p.clone() for n, p in block.moe.named_parameters()}
    rnd = randomize_like(block, seed=0)
    for n, p in rnd.moe.named_parameters():
        assert p.shape == before[n].shape
        assert not p.requires_grad
        if p.ndim >= 2:
            assert not torch.equal(p, before[n])
    for n, p in block.moe.named_parameters():  # original untouched
        assert torch.equal(p, before[n])
    rnd.record_routing = True  # hooks work on the copy
    with torch.no_grad():
        rnd(torch.randn(1, 4, model.config.hidden_size))
    assert len(rnd.router_calls) == 1


def test_capture_module_inputs_stops_early():
    from transplant.models.capture import run_until

    model = tiny_model_b()
    norms = {1: model.model.layers[1].post_attention_layernorm}
    with capture_module_inputs(norms, stop_after=1) as store:
        out = run_until(model, input_ids=torch.randint(4, 50, (1, 8)))
    assert out is None and 1 in store


def test_router_logits_recorded_when_router_forward_is_bypassed():
    """MobileMoE computes its router logits with F.linear on router.weight, so router.forward
    (and any hook on it) never runs; the logits come back as the block's second output.
    Routing diagnostics must still see them, through the extracted block and inside B's own
    forward pass."""
    model = tiny_model_b(seed=1)
    blocks = extract_blocks(model, TINY_B_CFG, [0, 1])
    for blk in blocks.values():
        blk.router._forward_hooks.clear()  # as if router.forward were bypassed
        blk.record_routing = True
    u = torch.randn(2, 5, model.config.hidden_size)
    with torch.no_grad():
        blocks[0](u)
        model.model(torch.randint(0, 50, (2, 7)))
    assert [tuple(c.shape) for c in blocks[0].router_calls] == [(10, 6), (14, 6)]
    assert [tuple(c.shape) for c in blocks[1].router_calls] == [(14, 6)]
