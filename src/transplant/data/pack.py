"""Pack documents into fixed-length sequences: <bos> + 1,023 tokens, documents separated by <eos>."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import numpy as np
import torch


class Packer:
    def __init__(self, bos_id: int, eos_id: int, seq_len: int, buffer: Optional[list[int]] = None):
        self.bos_id, self.eos_id, self.seq_len = bos_id, eos_id, seq_len
        self.buffer: list[int] = list(buffer or [])

    def add(self, ids: list[int]) -> None:
        self.buffer.extend(ids)
        self.buffer.append(self.eos_id)

    def ready(self) -> bool:
        return len(self.buffer) >= self.seq_len - 1

    def pop(self) -> list[int]:
        body = self.buffer[: self.seq_len - 1]
        del self.buffer[: self.seq_len - 1]
        return [self.bos_id] + body


class PackedStream:
    """Sequences from a TokenMixture. ``state_dict()`` covers the mixture and the leftover buffer."""

    def __init__(self, mixture, bos_id: int, eos_id: int, seq_len: int, buffer: Optional[list[int]] = None):
        self.mixture = mixture
        self.packer = Packer(bos_id, eos_id, seq_len, buffer)
        self._it = iter(mixture)

    def next_sequence(self) -> list[int]:
        while not self.packer.ready():
            _, ids = next(self._it)
            self.packer.add(ids)
        return self.packer.pop()

    def next_batch(self, batch_size: int) -> torch.Tensor:
        return torch.tensor([self.next_sequence() for _ in range(batch_size)], dtype=torch.long)

    def state_dict(self) -> dict:
        return {"mixture": self.mixture.state_dict(), "buffer": list(self.packer.buffer)}


def pack_token_lists(token_lists: Iterable[list[int]], bos_id: int, eos_id: int, seq_len: int) -> np.ndarray:
    """Pack finished token lists into an (N, seq_len) uint32 array; the tail that does not fill
    a whole sequence is dropped."""
    packer = Packer(bos_id, eos_id, seq_len)
    rows = []
    for ids in token_lists:
        packer.add(ids)
        while packer.ready():
            rows.append(packer.pop())
    if not rows:
        return np.zeros((0, seq_len), dtype=np.uint32)
    return np.asarray(rows, dtype=np.uint32)


def iterate_batches(array: np.ndarray, batch_size: int, max_rows: Optional[int] = None) -> Iterator[torch.Tensor]:
    n = array.shape[0] if max_rows is None else min(max_rows, array.shape[0])
    for start in range(0, n, batch_size):
        yield torch.from_numpy(array[start : min(start + batch_size, n)].astype(np.int64))
