"""Live progress bars for the long phases (tqdm, which ships with transformers).

Bars go to stderr and redraw in place every few seconds: tokens done, speed, time left and
the latest losses. Log lines are printed above the bars (``safe_log``) instead of breaking
them. Set TRANSPLANT_NO_PROGRESS=1 to switch the bars off (the log lines stay).
"""
from __future__ import annotations

import os
import sys
from typing import Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm comes with transformers
    tqdm = None


class _NoBar:
    def update(self, n=1) -> None:
        pass

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        pass

    def close(self) -> None:
        pass


def enabled() -> bool:
    return tqdm is not None and os.environ.get("TRANSPLANT_NO_PROGRESS") != "1"


def progress_bar(total: float, desc: str, unit: str = "tok", initial: float = 0, leave: bool = True):
    """A bar over ``total`` units (tokens by default), shown as 1.2M/5.0M with speed and ETA."""
    if not enabled():
        return _NoBar()
    return tqdm(total=total, initial=initial, desc=desc, unit=unit, unit_scale=True, leave=leave,
                mininterval=2.0, smoothing=0.05, file=sys.stderr, dynamic_ncols=True)


def safe_log(log: Callable) -> Callable:
    """``print`` replaced by a writer that keeps the progress bars intact; any other logger as is."""
    if log is print and enabled():
        return lambda msg="": tqdm.write(str(msg), file=sys.stdout)
    return log


class _SafeStream:
    """A stdout/stderr whose writes never fail. After Ctrl-C the reader of a pipe (`| tee`) may be
    gone or the terminal closed; output is then dropped instead of crashing a run that is about to
    save its checkpoint."""

    def __init__(self, stream):
        self._stream = stream
        self._broken = False

    def write(self, data):
        if not self._broken:
            try:
                return self._stream.write(data)
            except (OSError, ValueError):
                self._broken = True
        return len(data)

    def flush(self):
        if not self._broken:
            try:
                self._stream.flush()
            except (OSError, ValueError):
                self._broken = True

    def __getattr__(self, name):
        return getattr(self._stream, name)


def protect_output() -> None:
    """Wrap sys.stdout and sys.stderr once (call before any progress bar is made)."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, _SafeStream):
            setattr(sys, name, _SafeStream(stream))
