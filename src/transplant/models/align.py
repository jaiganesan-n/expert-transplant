"""Character-offset alignment between Gemma's and MobileMoE's tokenizers (plan A20).

A position pair (a, b) is kept only when both tokenizers end a token at the same
character offset: there, both models have read exactly the same text prefix.
Special tokens (empty offset spans) are never paired.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def aligned_pairs(offsets_a, offsets_b) -> list[tuple[int, int]]:
    """offsets_*: sequences of (start, end) character spans, one per token."""
    def last_token_per_end(offsets) -> dict[int, int]:
        # Byte-level tokenizers can give several tokens the same end offset (pieces of
        # one character); only the last has read the whole character.
        out: dict[int, int] = {}
        for idx, (start, end) in enumerate(offsets):
            if end > start:
                out[int(end)] = idx
        return out

    end_to_a = last_token_per_end(offsets_a)
    end_to_b = last_token_per_end(offsets_b)
    return [(end_to_a[e], end_to_b[e]) for e in sorted(end_to_a.keys() & end_to_b.keys())]


@dataclass
class AlignedBatch:
    ids_a: torch.Tensor  # (B, T_a)
    ids_b: torch.Tensor  # (B, T_b)
    mask_a: torch.Tensor
    mask_b: torch.Tensor
    pairs: list[list[tuple[int, int]]]  # per row
    n_tokens_a: int
    n_tokens_b: int

    def index_tensors(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, pos_a, pos_b = [], [], []
        for r, plist in enumerate(self.pairs):
            for a, b in plist:
                rows.append(r)
                pos_a.append(a)
                pos_b.append(b)
        mk = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
        return mk(rows), mk(pos_a), mk(pos_b)

    @property
    def n_pairs(self) -> int:
        return sum(len(p) for p in self.pairs)


def _encode(tok, texts, max_len):
    enc = tok(
        texts,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    return enc


def encode_aligned(tok_a, tok_b, texts: list[str], max_len: int, skip_first: bool = True) -> AlignedBatch:
    """Tokenize the same texts with both tokenizers and pair aligned positions.

    Padding is on the right; padded positions have empty offsets and are never paired.
    ``skip_first`` drops pairs at the first real token of either side, whose states
    behave like attention sinks.
    """
    for tok in (tok_a, tok_b):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    ea = _encode(tok_a, texts, max_len)
    eb = _encode(tok_b, texts, max_len)
    pairs = []
    for r in range(len(texts)):
        oa = ea["offset_mapping"][r].tolist()
        ob = eb["offset_mapping"][r].tolist()
        # Truncation can cut the two sides at different characters; pairs stay valid
        # because a shared end offset means the same prefix on both sides.
        p = aligned_pairs(oa, ob)
        if skip_first:
            first_a = next((i for i, (s, e) in enumerate(oa) if e > s), None)
            first_b = next((i for i, (s, e) in enumerate(ob) if e > s), None)
            p = [(a, b) for a, b in p if a != first_a and b != first_b]
        pairs.append(p)
    return AlignedBatch(
        ids_a=ea["input_ids"],
        ids_b=eb["input_ids"],
        mask_a=ea["attention_mask"],
        mask_b=eb["attention_mask"],
        pairs=pairs,
        n_tokens_a=int(ea["attention_mask"].sum()),
        n_tokens_b=int(eb["attention_mask"].sum()),
    )
