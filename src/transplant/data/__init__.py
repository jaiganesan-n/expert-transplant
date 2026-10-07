"""Data: the MobileMoE pre-training mixture (Table B.1), streamed, packed and held out."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .heldout import check_manifest, load_manifest
from .mixture import TokenMixture
from .pack import PackedStream
from .sources import DocSource, specs_from_cfg


def training_stream(data_cfg: dict, tok, state: Optional[dict] = None, source_factory=None, seed: Optional[int] = None) -> PackedStream:
    """The training token stream: every source minus its held-out head documents.

    ``seed`` selects which seeded 1-in-8 of each source's documents is skipped, so the
    seeds of a headline run (0, 1, 2) train on different data.
    """
    if source_factory is None:
        drop = int(data_cfg.get("seed_drop_one_in", 8))
        source_factory = lambda spec: DocSource(spec, subsample_seed=seed, drop_one_in=drop)  # noqa: E731
    manifest = load_manifest(data_cfg["heldout_dir"])
    check_manifest(manifest, data_cfg)
    specs = specs_from_cfg(data_cfg)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    mixture = TokenMixture(
        specs,
        encode,
        head_docs=manifest["head_docs"],
        state=(state or {}).get("mixture"),
        source_factory=source_factory,
    )
    return PackedStream(mixture, tok.bos_token_id, tok.eos_token_id, int(data_cfg["seq_len"]), (state or {}).get("buffer"))


def heldout_path(data_cfg: dict, name: str) -> Path:
    return Path(data_cfg["heldout_dir"]) / f"val_{name}.npy"
