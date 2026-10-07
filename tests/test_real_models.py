"""The gating tests on the real Gemma 3 1B and MobileMoE-M-Base (M2). Nothing trains until they pass.

Run on the Mac (M5, MPS) after `uv run hf auth login`, with both licenses accepted:

    TRANSPLANT_REAL=1 uv run pytest tests/test_real_models.py -v

Test 5 (memory) needs the GPU (MPS or CUDA). It measures in a fresh process
(`transplant memcheck`), so the models the other tests load cannot inflate it; between
tests, memory is handed back to macOS.
"""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from conftest import REPO, real_enabled
from transplant.config import load_config
from transplant.models import surgery
from transplant.models.load import build_hybrid, load_model_a, load_model_b_full, load_tokenizer
from transplant.models.moe_source import extract_blocks
from transplant.train.losses import chunked_ce_kl, shift_for_lm
from transplant.utils.device import pick_device, resolve_precision

pytestmark = [pytest.mark.real, pytest.mark.skipif(not real_enabled(), reason="set TRANSPLANT_REAL=1")]


@pytest.fixture(autouse=True)
def _free_memory_after_each_test():
    """Each test loads multi-GB models; collect them and empty the GPU cache before the next."""
    yield
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def cfg():
    return load_config([REPO / "configs" / "base.yaml"])  # all 26 layers, as in training


@pytest.fixture(scope="module")
def device():
    return pick_device()


def _text_ids(tok, device, n=2, length=256):
    text = ("The transplanted expert layer reads Gemma's hidden state through a learned projection. " * 40)
    ids = tok([text] * n, return_tensors="pt", truncation=True, max_length=length)["input_ids"]
    return ids.to(device)


def test_1_noop_surgery(cfg, device):
    """Gemma's own FFN behind identity projections, inside its sandwich norms, reproduces Gemma."""
    model = load_model_a(cfg.pair.a, torch.float32, device)
    tok = load_tokenizer(cfg.pair.a, "a")
    ids = _text_ids(tok, device)
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    d = model.config.hidden_size
    slots = {}
    for i, layer in enumerate(surgery.decoder_layers(model)):
        eye = lambda: nn.Linear(d, d, bias=False).to(device)  # noqa: E731
        w_in, w_out = eye(), eye()
        with torch.no_grad():
            w_in.weight.copy_(torch.eye(d))
            w_out.weight.copy_(torch.eye(d))
        slots[i] = surgery.ReplaceFFN(layer.mlp, copy.deepcopy(layer.mlp), w_in, w_out, layer_a=i, layer_b=i,
                                      b_dtype=torch.float32, proj_dtype=torch.float32)
    surgery.attach(model, slots)
    with torch.no_grad():
        out = model(input_ids=ids).logits
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_2_b_extraction(cfg, device):
    """B's extracted norm + MoE block, given B's own hidden states, reproduces B's MoE output."""
    policy = resolve_precision({"mode": "auto"}, device)
    model = load_model_b_full(cfg.pair.b, policy.b_dtype, device)
    tok = load_tokenizer(cfg.pair.b, "b")
    blocks = extract_blocks(model, cfg.pair.b, [0, 13, 25])
    ins, outs, hooks = {}, {}, []
    for j, blk in blocks.items():
        hooks.append(blk.norm.register_forward_pre_hook(lambda m, i, j=j: ins.__setitem__(j, i[0].detach())))
        hooks.append(blk.moe.register_forward_hook(
            lambda m, i, o, j=j: outs.__setitem__(j, (o[0] if isinstance(o, tuple) else o).detach())))
    for blk in blocks.values():  # routing diagnostics read B's router logits (MobileMoE bypasses router.forward)
        blk.record_routing = True
    ids = _text_ids(tok, device)
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    for j, blk in blocks.items():
        assert len(blk.router_calls) == 1 and tuple(blk.router_calls[0].shape) == (ids.numel(), 60)
        blk.record_routing = False
        with torch.no_grad():
            got = blk(ins[j])
        tol = 1e-2 if policy.b_dtype != torch.float32 else 1e-5
        assert got.shape == ins[j].shape  # MobileMoE returns (batch * seq, d); the block restores the shape
        torch.testing.assert_close(got.reshape(outs[j].shape).float(), outs[j].float(), rtol=tol, atol=tol)


def test_3_switch_off_equals_gemma(cfg, device):
    policy = resolve_precision({"mode": "auto"}, device)
    ref = load_model_a(cfg.pair.a, policy.a_dtype, device)
    tok = load_tokenizer(cfg.pair.a, "a")
    ids = _text_ids(tok, device)
    with torch.no_grad():
        want = ref(input_ids=ids).logits
    del ref
    hybrid = build_hybrid(cfg, device, policy)
    with torch.no_grad(), surgery.mode(hybrid.model, "original"):
        got = hybrid.model(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_4_gradient_isolation(cfg, device):
    policy = resolve_precision({"mode": "auto"}, device)
    hybrid = build_hybrid(cfg, device, policy)
    model = hybrid.model
    surgery.train_mode(model)
    ids = _text_ids(hybrid.tokenizer, device, n=1)
    h = model.model(input_ids=ids).last_hidden_state
    hs, labels = shift_for_lm(h, ids)
    ce, _, n = chunked_ce_kl(hs, labels, model.lm_head.weight, chunk=128)
    (ce / n).backward()
    named = [n for n, p in model.named_parameters() if p.grad is not None]
    assert named and all(".w_in." in n or ".w_out." in n for n in named)


@pytest.mark.gpu
@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="needs a GPU")
def test_5_memory_all_26_layers(tmp_path):
    """Real training steps with all 26 layers (local matching, then end to end with the teacher
    pass, chunked CE + KL and backward, each with an AdamW step) peak below the limit: the GPU
    working set macOS recommends for this Mac (about two thirds of its memory), or
    TRANSPLANT_MEM_LIMIT_GB. Measured in a fresh process by `transplant memcheck`."""
    import json
    import subprocess
    import sys

    out = tmp_path / "memcheck.json"
    proc = subprocess.run(
        [sys.executable, "-m", "transplant", "memcheck", "--config", str(REPO / "configs" / "base.yaml"),
         "--json", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=3600,
    )
    print(proc.stdout[-4000:])
    assert out.exists(), proc.stderr[-4000:]
    res = json.loads(out.read_text())
    print(f"peak {res['peak_gb']} GB (live tensors {res['peak_live_gb']} GB), weights {res['weights_gb']} GB, "
          f"limit {res['limit_gb']} GB; local {res['local']}, end to end {res['e2e']}")
    assert res["transplanted_layers"] == 26
    assert res["peak_gb"] < res["limit_gb"]


def test_6_chunked_loss_real_vocab(cfg, device):
    model = load_model_a(cfg.pair.a, torch.float32, device)
    tok = load_tokenizer(cfg.pair.a, "a")
    ids = _text_ids(tok, device, n=1, length=128)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, :-1].float()
        ref = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        h = model.model(input_ids=ids).last_hidden_state
        hs, labels = shift_for_lm(h, ids)
        ce, _, _ = chunked_ce_kl(hs, labels, model.lm_head.weight, chunk=32,
                                 softcap=model.config.final_logit_softcapping)
    torch.testing.assert_close(ce, ref, rtol=1e-4, atol=1e-2)


def test_7_alignment_share(cfg):
    from transplant.models.align import encode_aligned

    tok_a, tok_b = load_tokenizer(cfg.pair.a, "a"), load_tokenizer(cfg.pair.b, "b")
    texts = ["Pretrained expert layers may be portable between independently trained transformers. " * 20,
             "def f(x):\n    return x ** 2  # squares\n" * 20]
    ab = encode_aligned(tok_a, tok_b, texts, 2048, skip_first=False)
    share = ab.n_pairs / ab.n_tokens_a
    print(f"aligned share of Gemma tokens: {share:.1%}")
    assert share > 0.3
