"""Provider-facing Sentinel-2 client abstractions.

The class below keeps provider-specific API calls isolated from the rest of the
pipeline. Sentinel Hub and Earth Engine credentials should be supplied through
their standard environment variables or provider-specific config extensions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Iterable

from data_ingestion.schemas import DownloadedScene, SceneMetadata, SentinelQuery
from utils.exceptions import IngestionError
from utils.io import ensure_dir


class Sentinel2Client:
    """Sentinel-2 Level-1C ingestion client with retry-aware downloads."""

    def __init__(
        self,
        provider: str,
        output_root: str | Path,
        bands: Iterable[str],
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = provider
        self.output_root = Path(output_root)
        self.bands = list(bands)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.logger = logger or logging.getLogger("methane_pipeline.ingestion")

    def search(self, query: SentinelQuery) -> list[SceneMetadata]:
        """Search for matching scenes.

        This starter implementation returns a deterministic placeholder scene.
        Replace `_search_sentinel_hub` or `_search_earth_engine` with live API
        calls while keeping this method's return type stable.
        """

        self.logger.info("Searching %s scenes for region=%s", self.provider, query.region_id)
        tile_id = "T_PLACEHOLDER"
        return [
            SceneMetadata(
                timestamp=datetime.fromisoformat(query.date_start),
                cloud_coverage=min(query.max_cloud_coverage, 10.0),
                crs="EPSG:32611",
                tile_id=tile_id,
                region_id=query.region_id,
                provider=self.provider,
            )
        ]

    def download_scene(self, scene: SceneMetadata) -> DownloadedScene:
        """Download all configured bands for a scene with retry logic."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._download_scene_once(scene)
            except Exception as exc:  # noqa: BLE001 - convert provider errors to pipeline error
                last_error = exc
                self.logger.warning(
                    "Download failed for tile=%s attempt=%s/%s: %s",
                    scene.tile_id,
                    attempt,
                    self.max_retries,
                    exc,
                )
                sleep(self.retry_backoff_seconds * attempt)

        raise IngestionError(f"Failed to download scene {scene.tile_id}") from last_error

    def download_batch(self, scenes: Iterable[SceneMetadata]) -> list[DownloadedScene]:
        """Download many scenes."""

        return [self.download_scene(scene) for scene in scenes]

    def _download_scene_once(self, scene: SceneMetadata) -> DownloadedScene:
        date_part = scene.timestamp.date().isoformat()
        scene_dir = ensure_dir(self.output_root / scene.region_id / date_part / scene.tile_id)
        band_paths: dict[str, Path] = {}

        for band in self.bands:
            # Placeholder sidecar documents the expected target path until a live
            # provider implementation writes GeoTIFF assets here.
            band_path = scene_dir / f"{band}.tif"
            if not band_path.exists():
                band_path.touch()
            band_paths[band] = band_path

        metadata_path = scene_dir / "metadata.yaml"
        metadata_path.write_text(
            "\n".join(
                [
                    f"timestamp: {scene.timestamp.isoformat()}",
                    f"cloud_coverage: {scene.cloud_coverage}",
                    f"crs: {scene.crs}",
                    f"tile_id: {scene.tile_id}",
                    f"region_id: {scene.region_id}",
                    f"provider: {scene.provider}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        self.logger.info("Prepared scene directory %s", scene_dir)
        return DownloadedScene(metadata=scene, directory=scene_dir, band_paths=band_paths)
