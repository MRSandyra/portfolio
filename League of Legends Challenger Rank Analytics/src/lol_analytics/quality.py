"""Data-quality gates and compact acquisition manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TEN_MINUTES_MS, TWENTY_MINUTES_MS, VALID_TEAMS
from .data import read_events


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso_utc(timestamp_ms: int) -> str:
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).isoformat()


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _horizon_coverage(events: pd.DataFrame, cutoff_ms: int) -> dict[str, Any]:
    matches_reaching = 0
    matches_with_snapshot = 0
    eligible = 0
    lags: list[int] = []
    for _, match in events.groupby("matchId", sort=False):
        duration = int(match["timestamp"].max())
        reaches = duration >= cutoff_ms
        snapshots = match.loc[
            match["timestamp"].le(cutoff_ms) & match["totalGold_1"].notna(), "timestamp"
        ]
        has_snapshot = not snapshots.empty
        matches_reaching += int(reaches)
        matches_with_snapshot += int(has_snapshot)
        eligible += int(reaches and has_snapshot)
        if reaches and has_snapshot:
            lags.append(cutoff_ms - int(snapshots.max()))
    lag_array = np.asarray(lags, dtype=int)
    return {
        "cutoff_ms": cutoff_ms,
        "matches_reaching_horizon": matches_reaching,
        "matches_with_snapshot_at_or_before": matches_with_snapshot,
        "eligible_matches": eligible,
        "coverage_of_matches_reaching_horizon": (
            eligible / matches_reaching if matches_reaching else 0.0
        ),
        "snapshot_lag_ms": {
            "median": int(np.median(lag_array)) if lags else None,
            "p95": int(np.percentile(lag_array, 95)) if lags else None,
            "max": int(lag_array.max()) if lags else None,
        },
    }


def audit_events(events: pd.DataFrame, *, expected_matches: int | None = None) -> dict[str, Any]:
    grouped = events.groupby("matchId", sort=False)
    match_count = int(events["matchId"].nunique())
    end_counts = grouped["type"].apply(lambda values: int(values.eq("GAME_END").sum()))
    winner_counts = grouped["winningTeam"].apply(lambda values: values.dropna().nunique())
    winners = grouped["winningTeam"].first().astype(int)
    starts = grouped["gameStartTimestamp"].first().astype("int64")
    versions = grouped["gameVersion"].first().fillna("unknown").astype(str)
    patches = versions.str.extract(r"^(\d+\.\d+)", expand=False).fillna("unknown")

    duplicate_event_ids = None
    if "eventId" in events and events["eventId"].notna().any():
        duplicate_event_ids = int(events.duplicated(["matchId", "eventId"]).sum())

    critical_columns = [
        "matchId",
        "eventId",
        "gameStartTimestamp",
        "timestamp",
        "type",
        "winningTeam",
        "gameVersion",
        "platformId",
        "region",
    ]
    missingness = {
        column: float(events[column].isna().mean()) if column in events else 1.0
        for column in critical_columns
    }
    ten = _horizon_coverage(events, TEN_MINUTES_MS)
    twenty = _horizon_coverage(events, TWENTY_MINUTES_MS)

    gates = {
        "expected_match_count": expected_matches is None or match_count == expected_matches,
        "exactly_one_game_end_per_match": bool(end_counts.eq(1).all()),
        "one_winner_per_match": bool(winner_counts.eq(1).all()),
        "valid_winner_values": bool(winners.isin(VALID_TEAMS).all()),
        "unique_event_id_per_match": duplicate_event_ids in (None, 0),
        "ten_minute_snapshot_for_all_survivors": ten["eligible_matches"]
        == ten["matches_reaching_horizon"],
        "twenty_minute_snapshot_for_all_survivors": twenty["eligible_matches"]
        == twenty["matches_reaching_horizon"],
    }
    issues = [name for name, passed in gates.items() if not passed]
    return {
        "passed": not issues,
        "issues": issues,
        "gates": gates,
        "rows": int(len(events)),
        "matches": match_count,
        "event_types": _distribution(events["type"]),
        "winner_distribution": _distribution(winners),
        "patch_distribution": _distribution(patches),
        "region_distribution": _distribution(grouped["region"].first()),
        "platform_distribution": _distribution(grouped["platformId"].first()),
        "time_range_utc": {
            "start": _iso_utc(int(starts.min())),
            "end": _iso_utc(int(starts.max())),
        },
        "duplicate_event_ids": duplicate_event_ids,
        "critical_column_missing_fraction": missingness,
        "horizon_coverage": {"10m": ten, "20m": twenty},
    }


def audit_event_file(path: str | Path, *, expected_matches: int | None = None) -> dict[str, Any]:
    source = Path(path)
    events = read_events(source)
    audit = audit_events(events, expected_matches=expected_matches)
    audit["file"] = {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    audit["generated_at_utc"] = datetime.now(UTC).isoformat()
    return audit


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
