"""Wind and weather synchronization hooks."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def nearest_weather_record(timestamp: datetime, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the weather record nearest to an observation timestamp."""

    if not records:
        return None
    return min(records, key=lambda record: abs(timestamp - record["timestamp"]))
