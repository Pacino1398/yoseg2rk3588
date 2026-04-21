from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = REPO_ROOT / "yolo"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
TEST_INPUT_DIR = REPO_ROOT / "test_input"
WEIGHTS_DIR = REPO_ROOT / "weights"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def get_repo_root() -> Path:
    return REPO_ROOT


def get_yolo_dir() -> Path:
    return _env_path("YOSEGMENT_YOLO_DIR", YOLO_DIR)


def get_data_yaml() -> Path:
    return _env_path("YOSEGMENT_DATA_YAML", DATA_DIR / "my.yaml")


def get_runs_dir() -> Path:
    return _env_path("YOSEGMENT_RUNS_DIR", RUNS_DIR)


def get_test_input_dir() -> Path:
    return _env_path("YOSEGMENT_TEST_INPUT", TEST_INPUT_DIR)


def get_weights_dir() -> Path:
    return _env_path("YOSEGMENT_WEIGHTS_DIR", WEIGHTS_DIR)


def find_default_weights(preferred_name: str = "0414_qy++.pt") -> Path:
    weights_dir = get_weights_dir()
    preferred = weights_dir / preferred_name
    if preferred.exists():
        return preferred

    candidates = sorted(weights_dir.glob("*.pt"))
    if candidates:
        return candidates[0]

    return preferred


def get_default_mask_dir(run_name: str = "exp") -> Path:
    return get_runs_dir() / run_name / "masks"


def resolve_path(value: Optional[str | Path], default: Path) -> Path:
    if value is None:
        return default
    return Path(value).expanduser().resolve()
