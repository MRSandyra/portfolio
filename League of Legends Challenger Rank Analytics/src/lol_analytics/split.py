"""Deterministic absolute-time holdout splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_TIME_COLUMN = "game_start_timestamp_ms"


def _iso_utc(timestamp_ms: int) -> str:
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).isoformat()


@dataclass(frozen=True)
class HoldoutSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    order_column: str
    boundary_value: int

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": "absolute_time_holdout",
            "order_column": self.order_column,
            "timezone": "UTC",
            "boundary_timestamp_ms": self.boundary_value,
            "boundary_time_utc": _iso_utc(self.boundary_value),
            "train_range_utc": {
                "start": _iso_utc(int(self.train[self.order_column].min())),
                "end": _iso_utc(int(self.train[self.order_column].max())),
            },
            "test_range_utc": {
                "start": _iso_utc(int(self.test[self.order_column].min())),
                "end": _iso_utc(int(self.test[self.order_column].max())),
            },
        }


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    order_column: str = DEFAULT_TIME_COLUMN,
) -> HoldoutSplit:
    """Put the newest matches in test using a trusted absolute timestamp."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if order_column not in frame:
        raise ValueError(
            f"Missing trusted time column: {order_column}. "
            "Stream order is provenance only and cannot be used as chronology."
        )
    if frame[order_column].isna().any():
        raise ValueError(f"{order_column} contains missing values")
    if len(frame) < 4:
        raise ValueError("At least four matches are required for a holdout split")

    ordered = frame.sort_values([order_column, "matchId"], kind="stable").reset_index(drop=True)
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    test_size = min(test_size, len(ordered) - 2)
    boundary = int(ordered.iloc[-test_size][order_column])
    train = ordered.loc[ordered[order_column].lt(boundary)].copy()
    test = ordered.loc[ordered[order_column].ge(boundary)].copy()
    if len(train) < 2:
        raise ValueError(
            "At least two matches must precede the grouped absolute-time boundary"
        )
    if train[order_column].max() >= test[order_column].min():
        raise AssertionError("Internal error: absolute-time partitions overlap")
    return HoldoutSplit(
        train=train,
        test=test,
        order_column=order_column,
        boundary_value=boundary,
    )
