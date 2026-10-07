"""Streamed document sources: the open datasets MobileMoE-M was pre-trained on.

Nothing is downloaded in full. Each source is a Hugging Face dataset opened with
``streaming=True``, read document by document, and resumable from a saved state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


@dataclass
class SourceSpec:
    name: str
    weight: float
    domain: str = ""
    kind: str = "hf"  # hf | hf_files | synthetic
    hf_path: Optional[str] = None
    hf_name: Optional[str] = None
    split: str = "train"
    data_files: Optional[Any] = None  # for kind=hf_files (glob or list, hf:// paths)
    builder: str = "json"  # for kind=hf_files
    text_field: str = "text"
    formatter: str = "text"  # text | question_answer | chat
    max_chars: int = 32_000  # long documents are cut to bound tokenization time
    seed: int = 0  # synthetic only
    license: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, d: dict) -> "SourceSpec":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        return cls(**known, extra=extra)

    def identity(self) -> dict:
        """What defines the document stream (used in the held-out manifest hash)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "hf_path": self.hf_path,
            "hf_name": self.hf_name,
            "split": self.split,
            "data_files": self.data_files,
            "text_field": self.text_field,
            "formatter": self.formatter,
            "max_chars": self.max_chars,
            "seed": self.seed,
        }


# -- formatters ---------------------------------------------------------------

def _fmt_text(ex: dict, spec: SourceSpec) -> str:
    return ex.get(spec.text_field) or ""


def _fmt_question_answer(ex: dict, spec: SourceSpec) -> str:
    """Natural Reasoning: question, then the reference answer (or the first response)."""
    q = ex.get("question") or ""
    a = ex.get("reference_answer") or ""
    if not a:
        responses = ex.get("responses") or []
        if responses:
            first = responses[0]
            a = first.get("response", "") if isinstance(first, dict) else str(first)
    return f"{q}\n\n{a}".strip()


def _fmt_chat(ex: dict, spec: SourceSpec) -> str:
    """Nemotron post-training files: a list of input messages plus an output string."""
    parts = []
    msgs = ex.get("input") or ex.get("messages") or []
    if isinstance(msgs, str):
        parts.append(msgs)
    else:
        for m in msgs:
            parts.append(m.get("content", "") if isinstance(m, dict) else str(m))
    if ex.get("output"):
        parts.append(ex["output"])
    return "\n\n".join(p for p in parts if p)


FORMATTERS: dict[str, Callable[[dict, SourceSpec], str]] = {
    "text": _fmt_text,
    "question_answer": _fmt_question_answer,
    "chat": _fmt_chat,
}


# -- source --------------------------------------------------------------------

def _keep(name: str, index: int, seed: Optional[int], drop_one_in: int) -> bool:
    """Seeded document subsample: seed s drops a different 1-in-``drop_one_in`` of the documents."""
    if seed is None or drop_one_in <= 1:
        return True
    h = hashlib.blake2b(f"{name}:{index}:{seed}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % drop_one_in != 0


class DocSource:
    """Iterates the documents of one source, skipping a fixed head (held-out docs).

    ``state_dict()`` records how far the stream got; ``open(state=...)`` resumes there.
    With ``subsample_seed`` set, a seeded 1-in-``drop_one_in`` of the documents is
    skipped, so runs with different seeds read different data (held-out heads are
    always read whole).
    """

    def __init__(self, spec: SourceSpec, subsample_seed: Optional[int] = None, drop_one_in: int = 8):
        self.spec = spec
        self.docs_read = 0  # documents consumed since the head (not counting the head)
        self._ds = None
        self._head = 0
        self._pending_state: Optional[dict] = None
        self.subsample_seed = subsample_seed
        self.drop_one_in = drop_one_in

    def _base_dataset(self):
        from datasets import load_dataset

        s = self.spec
        if s.kind == "hf":
            return load_dataset(s.hf_path, s.hf_name, split=s.split, streaming=True)
        if s.kind == "hf_files":
            return load_dataset(s.builder, data_files=s.data_files, split=s.split, streaming=True)
        raise ValueError(f"source kind {s.kind} has no HF dataset")

    def open(self, skip_docs: int = 0, state: Optional[dict] = None) -> Iterator[str]:
        self._head = skip_docs
        fmt = FORMATTERS[self.spec.formatter]
        if self.spec.kind == "synthetic":
            from ..models.tiny import synthetic_doc

            start = skip_docs + (state or {}).get("docs", 0)
            self.docs_read = (state or {}).get("docs", 0)

            def gen():
                idx = start
                while True:
                    text = synthetic_doc(self.spec.seed, idx)
                    idx += 1
                    self.docs_read += 1
                    if _keep(self.spec.name, self.docs_read, self.subsample_seed, self.drop_one_in):
                        yield text[: self.spec.max_chars]

            return gen()

        base = self._base_dataset()
        ds = base.skip(skip_docs) if skip_docs else base
        self.docs_read = 0
        if state:
            self.docs_read = int(state.get("docs", 0))
            try:
                ds.load_state_dict(state["hf"])
                # Until the first document is read, the library reports its initial state;
                # keep the loaded one so a checkpoint taken before then stays correct.
                self._pending_state = state
            except Exception:
                # Fallback: count documents forward (slower, still exact).
                ds = base.skip(skip_docs + self.docs_read)
        self._ds = ds

        def gen():
            for ex in ds:
                self._pending_state = None
                self.docs_read += 1
                if not _keep(self.spec.name, self.docs_read, self.subsample_seed, self.drop_one_in):
                    continue
                text = fmt(ex, self.spec)
                if text:
                    yield text[: self.spec.max_chars]

        return gen()

    def state_dict(self) -> dict:
        if self._pending_state is not None:
            return dict(self._pending_state)
        st: dict[str, Any] = {"docs": self.docs_read}
        if self._ds is not None:
            try:
                st["hf"] = self._ds.state_dict()
            except Exception:
                pass
        return st


def specs_from_cfg(data_cfg: dict, include_zero: bool = False) -> list[SourceSpec]:
    specs = [SourceSpec.from_cfg(d) for d in data_cfg["sources"]]
    if not include_zero:
        specs = [s for s in specs if s.weight > 0]
    total = sum(s.weight for s in specs if s.weight > 0)
    if not include_zero and abs(total - 1.0) > 1e-6:
        for s in specs:
            s.weight = s.weight / total
    return specs
