"""Pausing, interruptions and one-run-at-a-time for long training runs.

**Pause.** Ctrl-C, closing the terminal window (SIGHUP), ``kill`` or a shutdown (SIGTERM), or a
file named ``PAUSE`` in the run folder all ask training to stop. Training stops at once, saves a
checkpoint at the last finished optimizer step and exits with ``PAUSED_EXIT``; running the same
command resumes from there with the same data, so nothing is lost but the step that was in
progress. The optimizer step and checkpoint writes are never cut in half: a request that arrives
during them takes effect right after. A third Ctrl-C quits at once (the last checkpoint is kept).

**Lock.** ``RunLock`` keeps a second process from training the same run folder while the first
one is still alive (for example while it saves its checkpoint after Ctrl-C).
"""
from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

PAUSED_EXIT = 75  # EX_TEMPFAIL: stopped on request; run the same command to resume
PAUSE_FILE = "PAUSE"


class StopRequested(BaseException):
    """Raised where training stands when a stop was asked for. A BaseException, so ordinary
    ``except Exception`` blocks do not swallow it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TrainingPaused(Exception):
    """Training saved a checkpoint and stopped on request."""

    def __init__(self, step: int, steps: int, reason: str, phase: str = "training"):
        where = f"step {step} of {steps}" if phase == "training" else phase
        super().__init__(f"paused in {where} ({reason})")
        self.step, self.steps, self.reason, self.phase = step, steps, reason, phase


class StopGuard:
    """Turns stop signals and the pause file into ``StopRequested`` at safe points.

    Outside a ``critical()`` block a signal raises at once; inside one it is remembered and raised
    when the block ends. ``check()`` raises if a stop is pending or the pause file exists.
    """

    SIGNALS = tuple(s for s in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP")) if s is not None)

    def __init__(self, pause_file: Optional[Path] = None, log=print):
        self.pause_file = Path(pause_file) if pause_file else None
        self.reason: Optional[str] = None
        self.presses = 0
        self._critical = 0
        self._old: dict = {}
        self._log = log

    def __enter__(self) -> "StopGuard":
        if threading.current_thread() is threading.main_thread():
            for s in self.SIGNALS:
                self._old[s] = signal.signal(s, self._handler)
        return self

    def __exit__(self, *exc) -> bool:
        for s, old in self._old.items():
            signal.signal(s, old)
        self._old.clear()
        return False

    def _handler(self, signum, frame) -> None:
        name = signal.Signals(signum).name
        self.presses += 1
        if self.presses >= 3:
            raise KeyboardInterrupt
        first = self.reason is None
        self.reason = self.reason or name
        if first:
            try:
                self._log(f"[stop] {name}: stopping and saving a checkpoint; run the same command to resume "
                          "(press Ctrl-C twice more to quit without saving)")
            except Exception:
                pass
        if self._critical == 0:
            raise StopRequested(self.reason)

    @contextmanager
    def critical(self):
        """A block a stop must not cut in half (optimizer step, checkpoint write)."""
        self._critical += 1
        try:
            yield
        finally:
            self._critical -= 1
        if self._critical == 0:
            self.check()

    @contextmanager
    def shielded(self):
        """Like ``critical`` but without raising afterwards (for the save that a stop triggers)."""
        self._critical += 1
        try:
            yield
        finally:
            self._critical -= 1

    def pause_requested(self) -> bool:
        return self.pause_file is not None and self.pause_file.exists()

    def check(self) -> None:
        if self.reason is None and self.pause_requested():
            self.reason = "pause file"
        if self.reason is not None:
            raise StopRequested(self.reason)

    def clear_pause_file(self) -> None:
        if self.pause_file is not None:
            try:
                self.pause_file.unlink()
            except FileNotFoundError:
                pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """``<run>/.lock`` holds the PID of the process training the run; a live holder blocks others."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / ".lock"
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid != os.getpid() and _alive(pid):
                    raise RuntimeError(
                        f"another process (pid {pid}) is still working on {self.path.parent}. If you just pressed "
                        "Ctrl-C, it is saving its checkpoint: wait a few seconds and run the command again. "
                        f"Otherwise stop it first (kill {pid}).")
                try:  # left behind by a process that is gone
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
            self.held = True
            return
        raise RuntimeError(f"could not take the lock {self.path}")

    def release(self) -> None:
        if self.held:
            try:
                if self.path.read_text().strip() == str(os.getpid()):
                    self.path.unlink()
            except OSError:
                pass
            self.held = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


def lock_holder(run_dir: Path) -> Optional[int]:
    """The PID training this run right now, or None."""
    p = Path(run_dir) / ".lock"
    try:
        pid = int(p.read_text().strip() or 0)
    except (OSError, ValueError):
        return None
    return pid if pid and _alive(pid) else None
