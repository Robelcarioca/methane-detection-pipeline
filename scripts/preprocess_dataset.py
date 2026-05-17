"""CLI: preprocess raw scenes into patch tensors."""

from __future__ import annotations

import argparse

from preprocessing.pipeline import run_preprocessing
from utils.config import load_config
from utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sentinel-2 scenes.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = configure_logging(config.section("paths").get("logs", "logs"))
    count = run_preprocessing(config.data)
    logger.info("Preprocessing complete: %s patch(es)", count)


if __name__ == "__main__":
    main()
