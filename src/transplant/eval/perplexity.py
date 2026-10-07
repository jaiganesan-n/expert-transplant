"""Held-out perplexity (Gemma tokens) and bits per byte (any tokenizer).

Per-token perplexities of Gemma and MobileMoE are not comparable (different
tokenizers), so B is reported in bits per byte, and A is too, for the side-by-side.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.pack import iterate_batches
from ..train.losses import chunked_ce_kl, shift_for_lm


def gemma_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    out = model.model(input_ids=ids, use_cache=False)
    return out.last_hidden_state


def lm_weight_and_softcap(model):
    return model.lm_head.weight, getattr(model.config, "final_logit_softcapping", None)


@torch.no_grad()
def lm_loss_on_array(
    model,
    array: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_rows: Optional[int] = None,
    chunk: int = 256,
    on_batch: Optional[Callable[[torch.Tensor], None]] = None,
) -> dict:
    """Mean next-token CE and top-1 accuracy over packed (N, T) Gemma-token sequences.
    Position 0 (<bos>) is input only."""
    weight, softcap = lm_weight_and_softcap(model)
    nll, count, correct = 0.0, 0, 0
    for batch in iterate_batches(array, batch_size, max_rows):
        ids = batch.to(device)
        h = gemma_hidden(model, ids)
        hs, labels = shift_for_lm(h, ids)
        ce_sum, _, n, c = chunked_ce_kl(hs, labels, weight, chunk=chunk, softcap=softcap, return_correct=True)
        nll += float(ce_sum)
        count += n
        correct += c
        if on_batch is not None:
            on_batch(ids)
    ce = nll / max(count, 1)
    return {"ce": ce, "ppl": math.exp(ce), "accuracy": correct / max(count, 1), "tokens": count}


def _pack_texts(tok, texts: list[str], seq_len: int):
    """Pack documents for one tokenizer; returns (sequences, target mask excluding <eos> targets)."""
    bos, eos = tok.bos_token_id, tok.eos_token_id
    seqs, buf = [], []
    for text in texts:
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= seq_len - 1:
            seqs.append([bos] + buf[: seq_len - 1])
            del buf[: seq_len - 1]
    if buf:
        seqs.append([bos] + buf)
    return seqs


@torch.no_grad()
def bits_per_byte(model, tok, texts: list[str], seq_len: int, device: torch.device, batch_size: int = 1, use_logits: bool = True) -> dict:
    """Total NLL (bits) of the documents' tokens divided by their UTF-8 bytes.

    ``use_logits=True`` calls the model's own forward (needed for MobileMoE's custom
    head); False uses Gemma's chunked head path. Separator <eos> targets are excluded.
    """
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    seqs = _pack_texts(tok, texts, seq_len)
    eos = tok.eos_token_id
    nll = 0.0
    n_tok = 0
    for k in range(0, len(seqs), batch_size):
        group = seqs[k : k + batch_size]
        for seq in group:  # variable final length: one at a time keeps it simple
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            labels = ids[:, 1:].clone()
            labels[labels == eos] = -100
            if use_logits:
                logits = model(input_ids=ids).logits[:, :-1].float()
                logp = F.log_softmax(logits, dim=-1)
                valid = labels >= 0
                tok_nll = -logp.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1)
                nll += float((tok_nll * valid).sum())
                n_tok += int(valid.sum())
            else:
                weight, softcap = lm_weight_and_softcap(model)
                h = gemma_hidden(model, ids)[:, :-1]
                ce_sum, _, n = chunked_ce_kl(h, labels, weight, softcap=softcap)
                nll += float(ce_sum)
                n_tok += n
    return {"bpb": nll / (math.log(2) * max(total_bytes, 1)), "bytes": total_bytes, "tokens": n_tok}
