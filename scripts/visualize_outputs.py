"""CLI: generate methane plume visualizations."""

from __future__ import annotations

import argparse

from utils.config import load_config
from utils.logging import configure_logging
from visualization.workflow import run_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate methane plume visualizations.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = configure_logging(config.section("paths").get("logs", "logs"))
    output = run_visualization(config.data)
    logger.info("Visualization written: %s", output)


if __name__ == "__main__":
    main()
