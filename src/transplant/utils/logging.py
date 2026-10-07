"""Run logging: metrics.jsonl per run, run_info.json, and one row per run in results/runs.csv.

Experiment tracking is open (plan question Q4). JSONL is the default; TensorBoard is
used too when it is installed and ``log.tensorboard: true``.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

RUNS_CSV_FIELDS = [
    "time",
    "run_name",
    "experiment",
    "config_hash",
    "git_commit",
    "seed",
    "layers",
    "stage",
    "tokens",
    "metric_set",
    "ppl_a",
    "ppl_ablated",
    "ppl_hybrid",
    "gap_ablated_minus_a",
    "gr",
    "gr_ce",
    "ppl_ratio",
    "acc_a",
    "acc_hybrid",
    "min_routing_entropy",
    "peak_gb",
    "tokens_per_s",
    "notes",
]


def git_commit(repo_dir: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


class RunLogger:
    def __init__(self, run_dir: str | Path, use_tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception:
                self._tb = None

    def log(self, step: int, stage: str, **metrics: Any) -> None:
        row = {"time": time.time(), "stage": stage, "step": step, **metrics}
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if self._tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb.add_scalar(f"{stage}/{key}", value, step)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()


def append_runs_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUNS_CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUNS_CSV_FIELDS})
