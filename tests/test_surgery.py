"""Gating tests 1, 3 and 4 on tiny models: no-op surgery, switch-off equality, gradient isolation."""
import copy

import torch
from torch import nn

from transplant.models import surgery
from transplant.models.load import build_hybrid, load_model_a
from transplant.models.tiny import tiny_model_a
from transplant.train.losses import chunked_ce_kl, shift_for_lm


def _ids(vocab, b=2, t=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, vocab, (b, t), generator=g)
    ids[:, 0] = 1  # <bos>
    return ids


def _identity(d):
    lin = nn.Linear(d, d, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(d))
    return lin


def test_noop_surgery_matches_gemma():
    """Test 1: A's own FFN in the transplant interface, identity projections, inside Gemma's
    sandwich norms, gives unmodified A's logits."""
    model = tiny_model_a(seed=0)
    ids = _ids(model.config.vocab_size)
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    d = model.config.hidden_size
    layers = surgery.decoder_layers(model)
    slots = {}
    for i, layer in enumerate(layers):
        own = copy.deepcopy(layer.mlp)
        slots[i] = surgery.ReplaceFFN(layer.mlp, own, _identity(d), _identity(d), layer_a=i, layer_b=i,
                                      b_dtype=torch.float32, proj_dtype=torch.float32)
    surgery.attach(model, slots)
    with torch.no_grad():
        out = model(input_ids=ids).logits
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    surgery.detach(model)
    assert not surgery.slots(model)


def test_switch_off_equals_gemma(tiny_cfg, cpu, fp32_policy):
    """Test 3: with the transplants switched off the hybrid is exactly Gemma (the KL teacher)."""
    ref_model = load_model_a(tiny_cfg.pair.a, torch.float32, cpu, "eager")
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    ids = _ids(ref_model.config.vocab_size)
    with torch.no_grad():
        ref = ref_model(input_ids=ids).logits
        with surgery.mode(hybrid.model, "original"):
            off = hybrid.model(input_ids=ids).logits
        on = hybrid.model(input_ids=ids).logits
    torch.testing.assert_close(off, ref, rtol=0, atol=0)
    assert (on - ref).abs().max() > 1e-4


def test_zero_mode_is_the_ablation(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    ids = _ids(hybrid.model.config.vocab_size)
    with torch.no_grad(), surgery.mode(hybrid.model, "zero"):
        zero = hybrid.model(input_ids=ids).logits
    ref = load_model_a(tiny_cfg.pair.a, torch.float32, cpu, "eager")
    for i in hybrid.a_layers:
        surgery.decoder_layers(ref)[i].mlp = type("Z", (nn.Module,), {"forward": lambda self, x: torch.zeros_like(x)})()
    with torch.no_grad():
        manual = ref(input_ids=ids).logits
    torch.testing.assert_close(zero, manual)


def test_gradient_isolation(tiny_cfg, cpu, fp32_policy):
    """Test 4: after one end-to-end backward, only projection parameters have gradients."""
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    model = hybrid.model
    surgery.train_mode(model)
    ids = _ids(model.config.vocab_size)
    with torch.no_grad(), surgery.mode(model, "original"):
        h_t = model.model(input_ids=ids).last_hidden_state[:, :-1]
    h_s = model.model(input_ids=ids).last_hidden_state
    hs, labels = shift_for_lm(h_s, ids)
    ce, kl, n = chunked_ce_kl(hs, labels, model.lm_head.weight, h_teacher=h_t, beta=1.0, chunk=16)
    ((ce + kl) / n).backward()
    with_grad = {name for name, p in model.named_parameters() if p.grad is not None}
    assert with_grad, "no gradients at all"
    assert all((".w_in." in n or ".w_out." in n) for n in with_grad), sorted(with_grad)[:5]
    expected = {n for n, p in model.named_parameters() if ".w_in." in n or ".w_out." in n}
    assert with_grad == expected


def test_frozen_moe_blocks_stay_in_eval_mode(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    surgery.train_mode(hybrid.model)
    for slot in surgery.slots(hybrid.model).values():
        assert not slot.inner.training and not slot.ffn_a.training
    assert hybrid.model.model.layers[0].training


def test_router_trainable_ablation(tiny_cfg, cpu, fp32_policy):
    tiny_cfg.transplant.router_trainable = True
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    names = {n for n, p in hybrid.model.named_parameters() if p.requires_grad}
    assert any(".inner.moe.gate." in n for n in names)
    assert not any("experts" in n for n in names)


def test_shared_expert_switch_changes_output(tiny_cfg, cpu, fp32_policy):
    hybrid = build_hybrid(tiny_cfg, cpu, fp32_policy)
    ids = _ids(hybrid.model.config.vocab_size)
    with torch.no_grad():
        with_shared = hybrid.model(input_ids=ids).logits
        surgery.set_shared_expert(hybrid.model, False)
        without = hybrid.model(input_ids=ids).logits
    assert (with_shared - without).abs().max() > 1e-6


def test_projection_state_roundtrip(tiny_cfg, cpu, fp32_policy):
    h1 = build_hybrid(tiny_cfg, cpu, fp32_policy)
    state = surgery.projection_state(h1.model)
    assert set(k.split(".", 1)[0] for k in state) == {str(i) for i in h1.a_layers}
    h2 = build_hybrid(tiny_cfg, cpu, fp32_policy)
    surgery.load_projection_state(h2.model, state)
    ids = _ids(h1.model.config.vocab_size)
    with torch.no_grad():
        torch.testing.assert_close(h1.model(input_ids=ids).logits, h2.model(input_ids=ids).logits)
