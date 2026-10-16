"""Load YAML project configuration with an environment override."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read the project YAML configuration once per process."""
    path = Path(os.getenv("VANTARA_CONFIG", ROOT / "config/config.yaml"))
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config
