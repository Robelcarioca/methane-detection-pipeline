"""YAML configuration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class AppConfig:
    """Thin wrapper around the loaded YAML config."""

    data: dict[str, Any]
    source_path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigurationError(f"Missing required config section: {name}")
        return value


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration file."""

    source_path = Path(path)
    if not source_path.exists():
        raise ConfigurationError(f"Config file does not exist: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigurationError("Top-level config must be a mapping.")

    return AppConfig(data=data, source_path=source_path)
