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


@torch.no_grad()
def greedy(model, tok, prompt: str, max_new_tokens: int, device) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    weight = model.lm_head.weight
    eos = tok.eos_token_id
    for _ in range(max_new_tokens):
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        nxt = (h.to(weight.dtype) @ weight.T).float().argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if eos is not None and int(nxt) == eos:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


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
