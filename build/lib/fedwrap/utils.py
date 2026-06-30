"""Lightweight utilities used across the package."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seeds(seed: int) -> np.random.Generator:
    """Return a new NumPy Generator seeded with *seed*."""
    return np.random.default_rng(seed)


@dataclass
class JsonlLogger:
    """Append-only JSONL logger for per-generation history."""
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        return
