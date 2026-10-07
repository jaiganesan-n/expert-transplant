"""Data pipeline: token-weighted mixing, determinism, exact resume, held-out disjointness."""
import json

import numpy as np
import pytest

from transplant.data import training_stream
from transplant.data.heldout import build_heldout, check_manifest, load_manifest
from transplant.data.mixture import TokenMixture
from transplant.data.pack import Packer, pack_token_lists
from transplant.data.sources import DocSource, SourceSpec
from transplant.models.tiny import tiny_tokenizer


def _encode(tok):
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


def _specs():
    return [
        SourceSpec(name="web", weight=0.62, kind="synthetic", seed=1),
        SourceSpec(name="math", weight=0.116, kind="synthetic", seed=2),
        SourceSpec(name="code", weight=0.10, kind="synthetic", seed=3),
        SourceSpec(name="knowledge", weight=0.10, kind="synthetic", seed=4),
        SourceSpec(name="science", weight=0.064, kind="synthetic", seed=5),
    ]


def test_token_shares_follow_weights():
    mix = TokenMixture(_specs(), _encode(tiny_tokenizer("a")))
    it = iter(mix)
    for _ in range(1500):
        next(it)
    shares = mix.shares()
    for s in _specs():
        assert abs(shares[s.name] - s.weight) < 0.01, (s.name, shares[s.name])


def test_mixture_is_deterministic_and_resumes_exactly():
    enc = _encode(tiny_tokenizer("a"))
    m1 = TokenMixture(_specs(), enc)
    it1 = iter(m1)
    first = [next(it1) for _ in range(50)]
    state = json.loads(json.dumps(m1.state_dict()))  # survives serialization
    rest_1 = [next(it1) for _ in range(50)]
    m2 = TokenMixture(_specs(), enc)
    it2 = iter(m2)
    assert [next(it2) for _ in range(50)] == first
    m3 = TokenMixture(_specs(), enc, state=state)
    it3 = iter(m3)
    assert [next(it3) for _ in range(50)] == rest_1


class LocalHFSource(DocSource):
    """A DocSource backed by a local Hugging Face IterableDataset (exercises the HF resume path)."""

    def _base_dataset(self):
        from datasets import Dataset

        texts = [f"document {self.spec.name} number {i} " + "word " * (i % 17) for i in range(400)]
        return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=4)


def test_hf_streaming_source_resumes_exactly():
    enc = _encode(tiny_tokenizer("a"))
    specs = [SourceSpec(name="a", weight=0.5, kind="hf"), SourceSpec(name="b", weight=0.5, kind="hf")]
    m1 = TokenMixture(specs, enc, head_docs={"a": 5, "b": 7}, source_factory=LocalHFSource)
    it1 = iter(m1)
    for _ in range(40):
        next(it1)
    state = m1.state_dict()
    rest = [next(it1) for _ in range(30)]
    m2 = TokenMixture(specs, enc, head_docs={"a": 5, "b": 7}, state=state, source_factory=LocalHFSource)
    it2 = iter(m2)
    assert [next(it2) for _ in range(30)] == rest


def test_packer_layout():
    p = Packer(bos_id=1, eos_id=2, seq_len=8)
    p.add([10, 11, 12])
    p.add([20, 21, 22, 23, 24])
    seq = p.pop()
    assert seq == [1, 10, 11, 12, 2, 20, 21, 22]
    arr = pack_token_lists([[5] * 20, [6] * 3], 1, 2, 8)
    assert arr.dtype == np.uint32 and arr.shape[1] == 8 and (arr[:, 0] == 1).all()


def test_heldout_is_never_trained_on(tiny_cfg, tmp_path):
    tok = tiny_tokenizer("a")
    manifest = build_heldout(tiny_cfg.data, tok, tiny_cfg.data.heldout_dir, log=lambda *a: None)
    held = set()
    for name in ("val_mix.jsonl", "calib.jsonl"):
        with open(f"{tiny_cfg.data.heldout_dir}/{name}") as fh:
            held |= {json.loads(line)["text"] for line in fh}
    stream = training_stream(tiny_cfg.data, tok)
    seen = []
    it = iter(stream.mixture)
    for _ in range(300):
        name, ids = next(it)
        seen.append(tok.decode(ids))
    # Synthetic docs are unique by index; decoded training docs never equal a held-out doc.
    norm = lambda s: s.replace(" ", "")  # noqa: E731
    held_n = {norm(t) for t in held}
    assert not any(norm(t) in held_n for t in seen)
    assert manifest["val_tokens"] > 0 and manifest["calib_docs"] > 0
    assert sum(manifest["head_docs"].values()) == sum(v["docs"] for v in manifest["per_source"].values())


def test_manifest_guards_against_a_changed_mixture(tiny_cfg):
    tok = tiny_tokenizer("a")
    build_heldout(tiny_cfg.data, tok, tiny_cfg.data.heldout_dir, log=lambda *a: None)
    manifest = load_manifest(tiny_cfg.data.heldout_dir)
    tiny_cfg.data.sources[0]["weight"] = 0.9
    with pytest.raises(RuntimeError):
        check_manifest(manifest, tiny_cfg.data)


def test_mixture_config_matches_table_b1():
    """Domain shares of the default mix equal Table B.1 of the MobileMoE paper."""
    from pathlib import Path

    from transplant.config import load_config

    cfg = load_config([Path(__file__).resolve().parents[1] / "configs" / "base.yaml"])
    shares = {}
    for s in cfg.data.sources:
        shares[s["domain"]] = shares.get(s["domain"], 0) + s["weight"]
    expected = {"web": 0.62, "math": 0.116, "code": 0.10, "knowledge": 0.10, "science": 0.064}
    for k, v in expected.items():
        assert abs(shares[k] - v) < 1e-6, (k, shares[k])
