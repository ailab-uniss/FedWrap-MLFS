"""Configuration loading utilities.

Reads a YAML experiment configuration and provides dotted-key access.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    return config


def get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Traverse *config* using a dotted key path (e.g. ``'evolution.pop_size'``).

    Returns *default* if any intermediate key is missing.
    """
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a value in *config* using a dotted key (e.g. ``'dataset.name'``).

    Creates intermediate dicts as needed.  The *value* is cast to int or
    float when possible, so ``'50'`` becomes ``50``.
    """
    # Auto-cast simple numeric/boolean strings
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            for typ in (int, float):
                try:
                    value = typ(value)
                    break
                except (ValueError, TypeError):
                    pass

    parts = dotted.split(".")
    cur = config
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


@dataclass(frozen=True)
class Paths:
    """Resolved output paths for a single run."""

    out_dir: Path

    @staticmethod
    def from_config(config: dict[str, Any]) -> "Paths":
        out_dir = Path(get(config, "logging.out_dir", "runs/unnamed"))
        return Paths(out_dir=out_dir)
