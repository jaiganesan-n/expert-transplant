"""Token-weighted mixture of document sources, deterministic and resumable.

Weights are shares of *tokens* (Table B.1 of the MobileMoE paper gives domain shares
of the pre-training mix). Each next document comes from the source furthest below its
target share so far ("deficit" scheduling). There is no randomness, so the same config
always yields the same token stream, and a run resumes exactly from a saved state.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from .sources import DocSource, SourceSpec


class TokenMixture:
    def __init__(
        self,
        specs: list[SourceSpec],
        encode: Callable[[str], list[int]],
        head_docs: Optional[dict[str, int]] = None,
        state: Optional[dict] = None,
        source_factory: Callable[[SourceSpec], DocSource] = DocSource,
    ):
        self.specs = [s for s in specs if s.weight > 0]
        if not self.specs:
            raise ValueError("mixture has no source with weight > 0")
        total = sum(s.weight for s in self.specs)
        self.targets = {s.name: s.weight / total for s in self.specs}
        self.encode = encode
        self.head_docs = head_docs or {}
        self.tokens = {s.name: 0 for s in self.specs}
        self.sources = {s.name: source_factory(s) for s in self.specs}
        src_states = {}
        if state:
            self.tokens.update(state.get("tokens", {}))
            src_states = state.get("sources", {})
        self._iters = {
            s.name: self.sources[s.name].open(self.head_docs.get(s.name, 0), src_states.get(s.name)) for s in self.specs
        }

    def _pick(self) -> str:
        total = sum(self.tokens.values()) + 1
        return max(self.specs, key=lambda s: self.targets[s.name] * total - self.tokens[s.name]).name

    def __iter__(self) -> Iterator[tuple[str, list[int]]]:
        while True:
            name = self._pick()
            try:
                text = next(self._iters[name])
            except StopIteration:
                raise RuntimeError(f"source '{name}' ran out of documents") from None
            ids = self.encode(text)
            self.tokens[name] += len(ids) + 1  # + EOS separator
            yield name, ids

    def shares(self) -> dict[str, float]:
        total = max(sum(self.tokens.values()), 1)
        return {k: v / total for k, v in self.tokens.items()}

    def state_dict(self) -> dict:
        return {
            "tokens": dict(self.tokens),
            "sources": {name: src.state_dict() for name, src in self.sources.items()},
        }
