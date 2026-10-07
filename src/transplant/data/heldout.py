"""Held-out data, built once and never trained on.

For every source the first documents of its stream become held-out data, in
proportion to the source's weight:

* validation: 5M Gemma tokens, packed and saved as ``val_mix.npy`` (plus raw text
  in ``val_mix.jsonl`` for bits-per-byte and routing agreement),
* calibration: 2M Gemma tokens of raw text in ``calib.jsonl`` for ridge init
  (A and B read it with their own tokenizers).

``manifest.json`` records how many head documents each source gave up; the training
stream skips exactly those, so training never sees held-out text. Two distribution-
shift sets (WikiText-103 test, C4 validation) are packed the same way.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .pack import pack_token_lists
from .sources import DocSource, SourceSpec, specs_from_cfg


def mixture_hash(data_cfg: dict) -> str:
    specs = specs_from_cfg(data_cfg)
    blob = json.dumps(
        {
            "sources": [dict(s.identity(), weight=round(s.weight, 8)) for s in specs],
            "heldout": {k: data_cfg["heldout"][k] for k in ("val_tokens", "calib_tokens")},
            "seq_len": data_cfg["seq_len"],
        },
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _encode_fn(tok):
    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    return encode


def _weighted_interleave(items: list[tuple[str, object, int]], weights: dict[str, float]) -> list[tuple[str, object]]:
    """items: (source, payload, n_tokens) in each source's stream order. Returns (source, payload)
    ordered so that, at every prefix, each source is as close as possible to its token share."""
    queues: dict[str, list] = {}
    for name, payload, n in items:
        queues.setdefault(name, []).append((payload, n))
    pos = {name: 0 for name in queues}
    taken = {name: 0 for name in queues}
    total_w = sum(weights.get(n, 0.0) for n in queues) or 1.0
    out = []
    while any(pos[n] < len(q) for n, q in queues.items()):
        total = sum(taken.values()) + 1
        live = [n for n in queues if pos[n] < len(queues[n])]
        name = max(live, key=lambda n: weights.get(n, 0.0) / total_w * total - taken[n])
        payload, n_tok = queues[name][pos[name]]
        pos[name] += 1
        taken[name] += n_tok
        out.append((name, payload))
    return out


def build_heldout(data_cfg: dict, tok, out_dir: str | Path, source_factory=DocSource, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = specs_from_cfg(data_cfg)
    encode = _encode_fn(tok)
    val_budget = int(data_cfg["heldout"]["val_tokens"])
    calib_budget = int(data_cfg["heldout"]["calib_tokens"])
    head_docs, per_source = {}, {}
    val_docs: list[tuple[str, str, list[int]]] = []
    calib_docs: list[tuple[str, str, int]] = []
    for spec in specs:
        src = source_factory(spec)
        it = src.open(0)
        n_docs = 0
        v_tok = c_tok = 0
        while v_tok < spec.weight * val_budget:
            text = next(it)
            ids = encode(text)
            val_docs.append((spec.name, text, ids))
            v_tok += len(ids) + 1
            n_docs += 1
        while c_tok < spec.weight * calib_budget:
            text = next(it)
            n_tok = len(encode(text)) + 1
            calib_docs.append((spec.name, text, n_tok))
            c_tok += n_tok
            n_docs += 1
        head_docs[spec.name] = src.docs_read  # includes empty docs the formatter dropped
        per_source[spec.name] = {"val_tokens": v_tok, "calib_tokens": c_tok, "docs": n_docs}
        log(f"[heldout] {spec.name}: {per_source[spec.name]}")

    # Interleave docs across sources by token share, so any prefix (eval subsets,
    # ridge.max_docs) is itself a mixture with the configured weights.
    weights = {spec.name: spec.weight for spec in specs}
    interleaved = [(n, t, ids) for n, (t, ids) in _weighted_interleave(
        [(n, (t, ids), len(ids) + 1) for n, t, ids in val_docs], weights)]
    seq_len = int(data_cfg["seq_len"])
    arr = pack_token_lists((ids for _, _, ids in interleaved), tok.bos_token_id, tok.eos_token_id, seq_len)
    np.save(out / "val_mix.npy", arr)
    with open(out / "val_mix.jsonl", "w") as fh:
        for name, text, _ in interleaved:
            fh.write(json.dumps({"source": name, "text": text}) + "\n")
    calib_mixed = _weighted_interleave([(n, t, k) for n, t, k in calib_docs], weights)
    with open(out / "calib.jsonl", "w") as fh:
        for name, text in calib_mixed:
            fh.write(json.dumps({"source": name, "text": text}) + "\n")
    manifest = {
        "mixture_hash": mixture_hash(data_cfg),
        "seq_len": seq_len,
        "tokenizer": getattr(tok, "name_or_path", "unknown"),
        "head_docs": head_docs,
        "per_source": per_source,
        "val_sequences": int(arr.shape[0]),
        "val_tokens": int(arr.size),
        "calib_docs": len(calib_docs),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"[heldout] val_mix: {arr.shape} sequences x tokens; calib docs: {len(calib_docs)}")
    return manifest


def build_shift_set(spec: dict, tok, out_dir: str | Path, seq_len: int, log=print) -> Optional[Path]:
    """WikiText-103 test (lines joined) or C4 validation (documents), packed with Gemma's tokenizer."""
    from datasets import load_dataset

    ds = load_dataset(spec["hf_path"], spec.get("hf_name"), split=spec["split"], streaming=True)
    encode = _encode_fn(tok)
    max_tokens = spec.get("max_tokens")
    lists, total = [], 0
    if spec.get("join_lines"):
        current: list[int] = []
        for ex in ds:
            line = ex[spec.get("text_field", "text")]
            if line.startswith(" = ") and not line.startswith(" = = ") and current:  # new article
                lists.append(current)
                total += len(current)
                current = []
            if line.strip():
                current.extend(encode(line))
            if max_tokens and total >= max_tokens:
                break
        if current:
            lists.append(current)
    else:
        for ex in ds:
            ids = encode(ex[spec.get("text_field", "text")])
            lists.append(ids)
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break
    arr = pack_token_lists(lists, tok.bos_token_id, tok.eos_token_id, seq_len)
    path = Path(out_dir) / f"val_{spec['name']}.npy"
    np.save(path, arr)
    log(f"[heldout] {spec['name']}: {arr.shape}")
    return path


def load_manifest(heldout_dir: str | Path) -> dict:
    path = Path(heldout_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m transplant build-heldout` first")
    with open(path) as fh:
        return json.load(fh)


def check_manifest(manifest: dict, data_cfg: dict) -> None:
    expected = mixture_hash(data_cfg)
    if manifest["mixture_hash"] != expected:
        raise RuntimeError(
            "held-out data was built for a different mixture "
            f"({manifest['mixture_hash']} != {expected}); rebuild it or restore the config"
        )


def read_jsonl_texts(path: str | Path, limit: Optional[int] = None) -> list[str]:
    texts = []
    with open(path) as fh:
        for k, line in enumerate(fh):
            if limit is not None and k >= limit:
                break
            texts.append(json.loads(line)["text"])
    return texts
