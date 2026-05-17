"""Typed data structures for satellite ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SentinelQuery:
    """Parameters used to search for Sentinel-2 scenes."""

    region_id: str
    date_start: str
    date_end: str
    bbox: tuple[float, float, float, float] | None = None
    polygon_path: Path | None = None
    max_cloud_coverage: float = 20.0


@dataclass(frozen=True)
class SceneMetadata:
    """Metadata tracked for each downloaded scene."""

    timestamp: datetime
    cloud_coverage: float
    crs: str
    tile_id: str
    region_id: str
    provider: str


@dataclass(frozen=True)
class DownloadedScene:
    """Local scene location and metadata."""

    metadata: SceneMetadata
    directory: Path
    band_paths: dict[str, Path]
