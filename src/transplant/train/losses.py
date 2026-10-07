"""Losses.

Local matching (per layer), normalized MSE pooled over the batch:

    L1 = sum_t ||W_out MoE_B(N_B(W_in x_t)) - FFN_A(x_t)||^2 / sum_t ||FFN_A(x_t)||^2

``local.loss: nmse_post_norm`` compares P_A(prediction) with P_A(target) instead,
because Gemma's post-FFN norm P_A discards the output's length (candidate ablation).

End to end: next-token cross-entropy through Gemma's head plus a KL term toward the
original Gemma:

    L2 = CE(y, p_hybrid) + beta * KL(p_A || p_hybrid)

Gemma's vocabulary has 262,144 entries, so a chunk of n positions has n x 262,144 logits
(0.27 GB in fp32 for n = 256). ``chunked_ce_kl`` works ``chunk`` positions at a time, and
its backward pass is written by hand (``_FusedCEKL``): it recomputes each chunk's logits
and turns them into the gradient in place,

    dL/dz = (g_ce + g_kl) softmax(z) - g_ce onehot(y) - g_kl p_A,

so at most two fp32 logit buffers per chunk are alive, instead of the five or six that
autograd keeps for log_softmax, KL and their gradients.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    return (pred - target).pow(2).sum() / target.pow(2).sum().clamp_min(eps)


def _logits(h: torch.Tensor, weight: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    # Matmul in the head's own dtype, then upcast the logits. Never copy the 262,144 x 1,152
    # head to fp32 (1.2 GB on a bf16 model); an fp32 head is used as is.
    logits = (h.to(weight.dtype) @ weight.T).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _ce_kl_forward(h_s, h_t, labels, weight, softcap, beta):
    """Token-summed CE, KL and top-1 hits for one chunk, without building a graph.

    Returns (ce, kl, correct, lse_student, lse_teacher or None). Peak memory: two fp32
    (n, V) buffers plus logsumexp's temporary.
    """
    valid = labels >= 0
    z = _logits(h_s, weight, softcap)  # (n, V), fp32
    lse = torch.logsumexp(z, dim=-1)
    z_y = z.gather(-1, labels.clamp_min(0)[:, None]).squeeze(-1)
    ce = ((lse - z_y) * valid).sum()
    correct = ((z.argmax(-1) == labels) & valid).sum().float()
    kl = torch.zeros((), device=z.device)
    lse_t = None
    if h_t is not None and beta > 0:
        zt = _logits(h_t, weight, softcap)
        lse_t = torch.logsumexp(zt, dim=-1)
        zt.sub_(lse_t[:, None])  # log p_A
        z.sub_(zt).neg_()  # log p_A - z (z is no longer needed)
        zt.exp_()  # p_A
        # KL = sum_v p_A (log p_A - z + lse) = sum_v p_A (log p_A - z) + lse
        rows = zt.mul_(z).sum(-1) + lse
        kl = (rows * valid).sum()
    return ce, kl, correct, lse, lse_t


class _FusedCEKL(torch.autograd.Function):
    """CE and KL for one chunk; backward recomputes the logits and forms dL/dz in place."""

    @staticmethod
    def forward(ctx, h_s, h_t, labels, weight, softcap, beta):
        with torch.no_grad():
            ce, kl, correct, lse, lse_t = _ce_kl_forward(h_s, h_t, labels, weight, softcap, beta)
        ctx.save_for_backward(h_s, h_t if lse_t is not None else None, labels, weight, lse, lse_t)
        ctx.softcap = softcap
        ctx.mark_non_differentiable(correct)
        return ce, kl, correct

    @staticmethod
    def backward(ctx, g_ce, g_kl, _g_correct):
        h_s, h_t, labels, weight, lse, lse_t = ctx.saved_tensors
        softcap = ctx.softcap
        valid = labels >= 0
        with torch.no_grad():
            z = (h_s.to(weight.dtype) @ weight.T).float()
            dcap = None
            if softcap:  # z = c tanh(u / c), dz/du = 1 - tanh^2
                z.div_(softcap).tanh_()
                dcap = 1.0 - z * z
                z.mul_(softcap)
            g = z.sub_(lse[:, None]).exp_()  # softmax(z), in place
            if h_t is not None:
                g.mul_(g_ce + g_kl)
                p = _logits(h_t, weight, softcap)
                p.sub_(lse_t[:, None]).exp_().mul_(g_kl)  # g_kl * p_A
                g.sub_(p)
                del p
            else:
                g.mul_(g_ce)
            # - g_ce at each row's target token; ignored rows get no gradient at all
            g.scatter_add_(1, labels.clamp_min(0)[:, None], (-g_ce * valid.to(g.dtype))[:, None])
            g.mul_(valid[:, None])
            if dcap is not None:
                g.mul_(dcap)
                del dcap
            g = g.to(weight.dtype)
            grad_h = (g @ weight).to(h_s.dtype) if ctx.needs_input_grad[0] else None
            grad_w = (g.T @ h_s.to(weight.dtype)) if ctx.needs_input_grad[3] else None
        return grad_h, None, None, grad_w, None, None


def chunked_ce_kl(
    h_student: torch.Tensor,
    labels: torch.Tensor,
    lm_weight: torch.Tensor,
    *,
    h_teacher: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    chunk: int = 256,
    softcap: Optional[float] = None,
    return_correct: bool = False,
):
    """Token-summed CE and KL over (N, D) final hidden states, ``chunk`` positions at a time.

    With gradients on, each chunk runs through ``_FusedCEKL``: only the chunk's hidden states
    are kept, and the backward pass recomputes its logits. Labels < 0 are ignored. Returns
    (ce_sum, kl_sum, n_valid), plus the number of correct top-1 predictions when
    ``return_correct`` is set.
    """
    h_student = h_student.reshape(-1, h_student.shape[-1])
    labels = labels.reshape(-1)
    if h_teacher is not None:
        h_teacher = h_teacher.reshape(-1, h_teacher.shape[-1]).detach()
    use_teacher = h_teacher is not None and beta > 0
    needs_grad = torch.is_grad_enabled() and (h_student.requires_grad or lm_weight.requires_grad)
    ce_total = torch.zeros((), device=h_student.device)
    kl_total = torch.zeros((), device=h_student.device)
    correct_total = torch.zeros((), device=h_student.device)
    for start in range(0, h_student.shape[0], chunk):
        sl = slice(start, start + chunk)
        ht = h_teacher[sl] if use_teacher else None
        if needs_grad:
            ce, kl, correct = _FusedCEKL.apply(h_student[sl], ht, labels[sl], lm_weight, softcap, float(beta))
        else:
            with torch.no_grad():
                ce, kl, correct, _, _ = _ce_kl_forward(h_student[sl], ht, labels[sl], lm_weight, softcap, beta)
        ce_total = ce_total + ce
        kl_total = kl_total + kl
        correct_total = correct_total + correct.detach()
    n_valid = int((labels >= 0).sum())
    if return_correct:
        return ce_total, kl_total, n_valid, int(correct_total)
    return ce_total, kl_total, n_valid


def reference_ce_kl(h_s, h_t, labels, weight, softcap=None, beta=0.0):
    """Plain autograd version of one chunk (tests compare ``chunked_ce_kl`` against it)."""
    logp_s = F.log_softmax(_logits(h_s, weight, softcap), dim=-1)
    valid = labels >= 0
    ce = -(logp_s.gather(-1, labels.clamp_min(0)[:, None]).squeeze(-1) * valid).sum()
    kl = torch.zeros((), device=h_s.device)
    if h_t is not None and beta > 0:
        with torch.no_grad():  # the teacher is a fixed target, also for a trainable head
            logp_t = F.log_softmax(_logits(h_t, weight, softcap), dim=-1)
        kl = ((logp_t.exp() * (logp_t - logp_s)).sum(-1) * valid).sum()
    return ce, kl


def shift_for_lm(hidden: torch.Tensor, input_ids: torch.Tensor, ignore_first_target: bool = False):
    """Predict token t+1 from position t. Returns (hidden[:, :-1], labels = ids[:, 1:])."""
    labels = input_ids[:, 1:].clone()
    if ignore_first_target:
        labels[:, 0] = -100
    return hidden[:, :-1], labels
