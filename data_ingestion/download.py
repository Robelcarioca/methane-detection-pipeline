"""High-level ingestion orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_ingestion.schemas import SentinelQuery
from data_ingestion.sentinel2_client import Sentinel2Client


def build_query(config: dict[str, Any]) -> SentinelQuery:
    """Build a Sentinel query from config."""

    query_config = config["ingestion"]["query"]
    bbox = query_config.get("bbox")
    return SentinelQuery(
        region_id=query_config["region_id"],
        date_start=query_config["date_start"],
        date_end=query_config["date_end"],
        bbox=tuple(bbox) if bbox else None,
        polygon_path=Path(query_config["polygon_path"]) if query_config.get("polygon_path") else None,
        max_cloud_coverage=float(query_config.get("max_cloud_coverage", 20)),
    )


def run_ingestion(config: dict[str, Any]) -> int:
    """Run search and batch download from a loaded config dictionary."""

    ingestion = config["ingestion"]
    bands = ingestion["bands"]["required"] + ingestion["bands"].get("optional_rgb", [])
    client = Sentinel2Client(
        provider=ingestion["provider"],
        output_root=config["paths"]["raw_data"],
        bands=bands,
        max_retries=int(ingestion.get("max_retries", 3)),
        retry_backoff_seconds=float(ingestion.get("retry_backoff_seconds", 5)),
    )
    scenes = client.search(build_query(config))
    downloads = client.download_batch(scenes)
    return len(downloads)
