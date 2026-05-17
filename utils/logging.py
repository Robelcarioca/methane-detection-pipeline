"""Logging setup for command-line and batch workflows."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging once."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("methane_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
