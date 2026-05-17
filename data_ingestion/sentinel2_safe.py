"""Sentinel-2 SAFE discovery helpers.

The SAFE product layout is nested and varies slightly between products. These
helpers keep path handling in one place so processing code can work with named
bands instead of directory assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class Sentinel2SafeError(RuntimeError):
    """Raised when a SAFE product is missing required structure or bands."""


@dataclass(frozen=True)
class Sentinel2BandSet:
    """Resolved Sentinel-2 bands for a single SAFE scene."""

    safe_path: Path
    red: Path
    nir: Path

    @property
    def scene_id(self) -> str:
        """Scene identifier safe for output directory names."""

        return self.safe_path.stem


def discover_safe_products(raw_root: str | Path) -> list[Path]:
    """Return Sentinel-2 SAFE product directories below ``raw_root``."""

    root = Path(raw_root)
    if not root.exists():
        raise Sentinel2SafeError(f"Raw data directory does not exist: {root}")

    products = sorted(path for path in root.rglob("*.SAFE") if path.is_dir())
    if not products:
        raise Sentinel2SafeError(f"No Sentinel-2 .SAFE products found under: {root}")
    return products


def _single_match(candidates: list[Path], band_name: str, safe_path: Path) -> Path:
    if not candidates:
        raise Sentinel2SafeError(f"Missing required Sentinel-2 band {band_name} in {safe_path}")
    if len(candidates) > 1:
        r10m = [path for path in candidates if "R10m" in path.parts]
        if len(r10m) == 1:
            return r10m[0]
        names = ", ".join(str(path) for path in candidates[:5])
        raise Sentinel2SafeError(f"Ambiguous {band_name} band in {safe_path}: {names}")
    return candidates[0]


def resolve_r10m_vegetation_bands(safe_path: str | Path) -> Sentinel2BandSet:
    """Resolve Sentinel-2 10 m red (B04) and NIR (B08) JP2 files."""

    safe = Path(safe_path)
    if not safe.exists() or not safe.is_dir():
        raise Sentinel2SafeError(f"SAFE product does not exist or is not a directory: {safe}")

    img_roots = list(safe.glob("GRANULE/*/IMG_DATA/R10m"))
    if not img_roots:
        raise Sentinel2SafeError(f"Missing GRANULE/*/IMG_DATA/R10m directory in {safe}")

    red_candidates: list[Path] = []
    nir_candidates: list[Path] = []
    for img_root in img_roots:
        red_candidates.extend(sorted(img_root.glob("*_B04_10m.jp2")))
        nir_candidates.extend(sorted(img_root.glob("*_B08_10m.jp2")))

    return Sentinel2BandSet(
        safe_path=safe,
        red=_single_match(red_candidates, "B04_10m", safe),
        nir=_single_match(nir_candidates, "B08_10m", safe),
    )
