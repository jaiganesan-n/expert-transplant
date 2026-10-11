"""Sample outputs: greedy continuations of fixed prompts from Gemma and from the hybrid.

Loss and accuracy say how much the transplant recovers; reading the continuations side
by side shows what it gets wrong. Written to ``samples.md`` in the run folder.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..models import surgery

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "def fibonacci(n):",
    "The derivative of x^2 with respect to x is",
    "In 1969, the first humans to walk on the Moon were",
]


SAMPLE_LENGTH = 128  # every greedy step runs on this many positions (see greedy)


@torch.no_grad()
def greedy(model, tok, prompt: str, max_new_tokens: int, device, length: int = SAMPLE_LENGTH) -> str:
    """Greedy decoding without a KV cache, on a fixed-length buffer.

    The sequence sits at the start of a ``length``-long buffer padded on the right; the model
    is causal, so the padding never affects the positions before it, and every step reads
    the logits at the last real position. One shape for every step and prompt: on the Apple
    GPU each new shape compiles new kernels that stay in memory.
    """
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
    n_new = max(0, min(max_new_tokens, length - len(ids)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    buf = torch.full((1, max(length, len(ids))), pad, dtype=torch.long, device=device)
    buf[0, : len(ids)] = torch.tensor(ids, device=device)
    weight = model.lm_head.weight
    eos = tok.eos_token_id
    cur = len(ids)
    for _ in range(n_new):
        h = model.model(input_ids=buf, use_cache=False).last_hidden_state[:, cur - 1]
        nxt = int((h.to(weight.dtype) @ weight.T).float().argmax(-1))
        buf[0, cur] = nxt
        cur += 1
        if eos is not None and nxt == eos:
            break
    return tok.decode(buf[0, :cur].tolist(), skip_special_tokens=True)


def write_samples(hybrid, cfg: dict, device, path: Path, label: str = "") -> list[dict]:
    ecfg = cfg["eval"]
    prompts = list(ecfg.get("sample_prompts") or DEFAULT_PROMPTS)
    n_new = int(ecfg.get("sample_tokens", 40))
    model, tok = hybrid.model, hybrid.tokenizer
    was_training = model.training
    model.eval()
    rows = []
    for prompt in prompts:
        with surgery.mode(model, "original"):
            a = greedy(model, tok, prompt, n_new, device)
        with surgery.mode(model, "transplant"):
            h = greedy(model, tok, prompt, n_new, device)
        rows.append({"prompt": prompt, "gemma": a, "hybrid": h})
    lines = [f"# Sample outputs {label}".rstrip(), "", f"Greedy, {n_new} new tokens. Gemma = original; hybrid = FFNs replaced.", ""]
    for r in rows:
        lines += [f"## {r['prompt']}", "", f"**Gemma:** {r['gemma']}", "", f"**Hybrid:** {r['hybrid']}", ""]
    path.write_text("\n".join(lines))
    if was_training:
        surgery.train_mode(model)
    return rows
