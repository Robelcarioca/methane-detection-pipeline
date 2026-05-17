"""Geospatial helpers shared by ingestion and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box in lon/lat order."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float, float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("BoundingBox requires four values: min_lon, min_lat, max_lon, max_lat")
        return cls(*map(float, values))


def crs_to_string(crs: Any) -> str:
    """Return a stable CRS string from rasterio/pyproj-like CRS objects."""

    if crs is None:
        return ""
    if hasattr(crs, "to_string"):
        return crs.to_string()
    return str(crs)
