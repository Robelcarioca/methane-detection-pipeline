"""Temporal matching utilities for multi-temporal observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TemporalPair:
    """Matched previous and current observation IDs."""

    previous_id: str
    current_id: str
    delta: timedelta


def nearest_temporal_pairs(
    observations: dict[str, datetime],
    max_delta: timedelta,
) -> list[TemporalPair]:
    """Find nearest prior observation for each timestamp."""

    ordered = sorted(observations.items(), key=lambda item: item[1])
    pairs: list[TemporalPair] = []
    for index, (current_id, current_time) in enumerate(ordered[1:], start=1):
        previous_id, previous_time = min(
            ordered[:index],
            key=lambda item: abs(current_time - item[1]),
        )
        delta = current_time - previous_time
        if timedelta(0) <= delta <= max_delta:
            pairs.append(TemporalPair(previous_id=previous_id, current_id=current_id, delta=delta))
    return pairs
