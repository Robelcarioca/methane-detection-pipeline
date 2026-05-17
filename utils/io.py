"""Filesystem and array IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    """Save a NumPy array, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)


def metadata_to_jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert common metadata values to JSON-friendly strings."""

    return {key: str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value for key, value in metadata.items()}
