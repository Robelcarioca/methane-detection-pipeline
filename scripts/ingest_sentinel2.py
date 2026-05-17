"""CLI: query and download Sentinel-2 imagery."""

from __future__ import annotations

import argparse

from data_ingestion.download import run_ingestion
from utils.config import load_config
from utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 Level-1C imagery.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = configure_logging(config.section("paths").get("logs", "logs"))
    count = run_ingestion(config.data)
    logger.info("Ingestion complete: %s scene(s)", count)


if __name__ == "__main__":
    main()
