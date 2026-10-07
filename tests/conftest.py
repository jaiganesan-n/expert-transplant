import os
from pathlib import Path

import pytest
import torch

from transplant.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_cfg(tmp_path):
    """base.yaml + the tiny pair, with every output under a temp dir."""
    return load_config(
        [REPO / "configs" / "base.yaml", REPO / "configs" / "pairs" / "tiny.yaml"],
        [
            f"run.out_dir={tmp_path / 'runs'}",
            f"data.heldout_dir={tmp_path / 'heldout'}",
            f"log.runs_csv={tmp_path / 'runs.csv'}",
        ],
    )


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture
def fp32_policy():
    from transplant.utils.device import PrecisionPolicy

    return PrecisionPolicy(torch.float32, torch.float32, torch.float32, False)


def real_enabled() -> bool:
    return os.environ.get("TRANSPLANT_REAL") == "1"
