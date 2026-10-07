"""Gating test 6: chunked cross-entropy and KL (hand-written backward) match the unchunked values and gradients."""
import pytest
import torch
import torch.nn.functional as F

from transplant.train.losses import chunked_ce_kl, nmse, reference_ce_kl


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_chunked_matches_unchunked(softcap):
    torch.manual_seed(0)
    n, d, v = 70, 16, 300
    hs = torch.randn(n, d, requires_grad=True)
    ht = torch.randn(n, d)
    w = torch.randn(v, d)
    labels = torch.randint(0, v, (n,))
    labels[5] = -100

    def logits(h):
        z = h @ w.T
        return torch.tanh(z / softcap) * softcap if softcap else z

    ls, lt = F.log_softmax(logits(hs), -1), F.log_softmax(logits(ht), -1)
    valid = labels >= 0
    ce_ref = F.cross_entropy(logits(hs), labels, ignore_index=-100, reduction="sum")
    kl_ref = ((lt.exp() * (lt - ls)).sum(-1) * valid).sum()
    (ce_ref + 0.5 * kl_ref).backward()
    g_ref = hs.grad.clone()
    hs.grad = None

    ce, kl, count = chunked_ce_kl(hs, labels, w, h_teacher=ht, beta=0.5, chunk=16, softcap=softcap)
    (ce + 0.5 * kl).backward()
    assert count == int(valid.sum())
    torch.testing.assert_close(ce, ce_ref, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(kl, kl_ref, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(hs.grad, g_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("w_ce,w_kl,beta,teacher", [(1.0, 0.0, 0.0, False), (0.3, 2.0, 1.0, True), (1.0, 1.0, 0.0, True)])
def test_fused_backward_matches_autograd(softcap, w_ce, w_kl, beta, teacher):
    """Every gradient path of the hand-written backward: CE only, CE and KL with different
    weights, a teacher given with beta = 0, ignored labels, a chunk with no valid label,
    and a trainable head (grad for the weight too)."""
    torch.manual_seed(1)
    n, d, v = 45, 12, 97
    hs = torch.randn(n, d, requires_grad=True)
    ht = torch.randn(n, d) if teacher else None
    w = torch.randn(v, d, requires_grad=True)
    labels = torch.randint(0, v, (n,))
    labels[3] = -100
    labels[32:40] = -100  # chunk 32..39 has no valid label

    ce_r, kl_r = reference_ce_kl(hs, ht, labels, w, softcap, beta)
    (w_ce * ce_r + w_kl * kl_r).backward()
    gh_ref, gw_ref = hs.grad.clone(), w.grad.clone()
    hs.grad = w.grad = None

    ce, kl, _ = chunked_ce_kl(hs, labels, w, h_teacher=ht, beta=beta, chunk=8, softcap=softcap)
    (w_ce * ce + w_kl * kl).backward()
    torch.testing.assert_close(ce, ce_r, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(kl, kl_r, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(hs.grad, gh_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w.grad, gw_ref, rtol=1e-4, atol=1e-5)


def test_bf16_head_gives_bf16_free_fp32_loss():
    """A bf16 head (as on the M5) with bf16 hidden states: finite fp32 sums, bf16 gradient."""
    torch.manual_seed(2)
    hs = torch.randn(20, 8, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, 8, dtype=torch.bfloat16)
    ce, kl, n, correct = chunked_ce_kl(hs, torch.randint(0, 64, (20,)), w, h_teacher=hs.detach() + 0.1, beta=1.0,
                                       chunk=6, return_correct=True)
    (ce + kl).backward()
    assert ce.dtype == torch.float32 and torch.isfinite(ce) and torch.isfinite(kl)
    assert hs.grad.dtype == torch.bfloat16 and torch.isfinite(hs.grad.float()).all()
    assert n == 20 and 0 <= correct <= 20


def test_kl_zero_when_teacher_equals_student():
    h = torch.randn(10, 8)
    w = torch.randn(50, 8)
    _, kl, _ = chunked_ce_kl(h, torch.randint(0, 50, (10,)), w, h_teacher=h, beta=1.0, chunk=4)
    assert abs(float(kl)) < 1e-5


def test_nmse_is_one_for_zero_prediction():
    y = torch.randn(4, 6)
    assert torch.isclose(nmse(torch.zeros_like(y), y), torch.tensor(1.0))
    assert float(nmse(y, y)) == 0.0
