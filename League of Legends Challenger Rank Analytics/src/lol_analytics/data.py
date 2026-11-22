"""Input loading, canonicalisation, and optional Hugging Face acquisition."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .config import VALID_TEAMS

BASE_COLUMNS = [
    "matchId",
    "eventId",
    "match_order",
    "gameStartTimestamp",
    "timestamp",
    "type",
    "killerTeamId",
    "participantId",
    "victimId",
    "killerId",
    "winningTeam",
    "monsterType",
    "monsterSubType",
    "buildingType",
    "towerType",
    "teamId",
    "gameVersion",
    "platformId",
    "region",
]
STAT_PREFIXES = (
    "totalGold",
    "xp",
    "minionsKilled",
    "jungleMinionsKilled",
    "championId",
    "level",
)
STAT_COLUMNS = [f"{prefix}_{slot}" for prefix in STAT_PREFIXES for slot in range(1, 11)]
SELECTED_COLUMNS = BASE_COLUMNS + STAT_COLUMNS


def read_events(path: str | Path) -> pd.DataFrame:
    """Read canonical event rows from JSONL, CSV, or Parquet."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, lines=True)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError(f"Unsupported input format: {source.suffix}")
    return canonicalize_events(frame)


def canonicalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and type the subset needed by feature engineering."""
    required = {"matchId", "timestamp", "type", "winningTeam", "gameStartTimestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required event columns: {sorted(missing)}")

    result = frame.copy()
    for column in [
        "timestamp",
        "eventId",
        "match_order",
        "gameStartTimestamp",
        "killerTeamId",
        "participantId",
        "victimId",
        "killerId",
        "winningTeam",
        "teamId",
        *STAT_COLUMNS,
    ]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    starts_per_match = result.groupby("matchId", sort=False)["gameStartTimestamp"].agg(
        lambda values: values.dropna().nunique()
    )
    missing_start = starts_per_match[starts_per_match.eq(0)].index.tolist()
    inconsistent_start = starts_per_match[starts_per_match.gt(1)].index.tolist()
    if missing_start:
        raise ValueError(
            "gameStartTimestamp is required for every match; reacquire data with the current schema. "
            f"Missing matches: {missing_start[:5]}"
        )
    if inconsistent_start:
        raise ValueError(
            "gameStartTimestamp must be constant within each match. "
            f"Inconsistent matches: {inconsistent_start[:5]}"
        )
    starts = result["gameStartTimestamp"].dropna()
    if not starts.between(1_000_000_000_000, 9_999_999_999_999).all():
        raise ValueError("gameStartTimestamp must be Unix epoch milliseconds")

    if "match_order" not in result:
        order = {match_id: index for index, match_id in enumerate(pd.unique(result["matchId"]))}
        result["match_order"] = result["matchId"].map(order)

    winners = set(result["winningTeam"].dropna().astype(int).unique())
    invalid = winners.difference(VALID_TEAMS)
    if invalid:
        raise ValueError(f"Unexpected winningTeam values: {sorted(invalid)}")
    return result.sort_values(
        ["gameStartTimestamp", "matchId", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def _slim_row(row: Mapping[str, Any]) -> dict[str, Any]:
    slim = {column: row.get(column) for column in SELECTED_COLUMNS if column in row}
    return slim


def collect_complete_matches(
    rows: Iterable[Mapping[str, Any]], limit: int, *, skip: int = 0
) -> pd.DataFrame:
    """Buffer stream rows and keep only matches terminated by ``GAME_END``."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    buffers: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    completed: list[list[dict[str, Any]]] = []
    completed_ids: set[Any] = set()
    completion_count = 0

    for row in rows:
        match_id = row.get("matchId")
        if match_id is None or match_id in completed_ids:
            continue
        buffers[match_id].append(_slim_row(row))
        if row.get("type") == "GAME_END":
            match_rows = buffers.pop(match_id)
            match_order = completion_count
            completion_count += 1
            completed_ids.add(match_id)
            if match_order < skip:
                continue
            for match_row in match_rows:
                match_row["match_order"] = match_order
            completed.append(match_rows)
            if len(completed) >= limit:
                break

    if len(completed) < limit:
        raise ValueError(
            f"Only {len(completed)} complete matches ending in GAME_END were found "
            f"after skipping {skip}; requested {limit}"
        )
    return canonicalize_events(pd.DataFrame(event for match in completed for event in match))


def collect_complete_matches_from_batches(
    batches: Iterable[Mapping[str, Sequence[Any]]], limit: int, *, skip: int = 0
) -> pd.DataFrame:
    """Collect complete matches from columnar batches without per-event Python dicts."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    buffers: dict[Any, list[pd.DataFrame]] = defaultdict(list)
    completed: list[pd.DataFrame] = []
    completed_ids: set[Any] = set()
    completion_count = 0

    for batch in batches:
        frame = pd.DataFrame(batch).reset_index(drop=True)
        if frame.empty or "matchId" not in frame or "type" not in frame:
            continue
        completion_order: list[tuple[int, Any]] = []
        for match_id, group in frame.groupby("matchId", sort=False, dropna=True):
            if match_id in completed_ids:
                continue
            end_indices = group.index[group["type"].eq("GAME_END")]
            if len(end_indices):
                end_index = int(end_indices[0])
                group = group.loc[group.index <= end_index]
                completion_order.append((end_index, match_id))
            buffers[match_id].append(group.copy())

        for _, match_id in sorted(completion_order):
            match_frame = pd.concat(buffers.pop(match_id), ignore_index=True)
            match_order = completion_count
            completion_count += 1
            completed_ids.add(match_id)
            if match_order < skip:
                continue
            match_frame["match_order"] = match_order
            completed.append(match_frame)
            if len(completed) >= limit:
                return canonicalize_events(pd.concat(completed, ignore_index=True))

    raise ValueError(
        f"Only {len(completed)} complete matches ending in GAME_END were found "
        f"after skipping {skip}; requested {limit}"
    )


def stream_huggingface_rows(
    dataset: str, split: str, config: str = "events"
) -> Iterator[Mapping[str, Any]]:
    """Yield remote rows lazily; importing ``datasets`` is intentionally optional."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'acquisition' extra to download Hugging Face data") from exc
    stream = load_dataset(dataset, name=config, split=split, streaming=True)
    available = set(stream.column_names or [])
    projected = [column for column in SELECTED_COLUMNS if column in available]
    if not projected:
        raise RuntimeError("The remote dataset exposes none of the required columns")
    yield from stream.select_columns(projected)


def stream_huggingface_batches(
    dataset: str,
    split: str,
    *,
    config: str = "events",
    batch_size: int = 10_000,
) -> Iterator[Mapping[str, Sequence[Any]]]:
    """Yield projected remote rows as Arrow-backed columnar batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'acquisition' extra to download Hugging Face data") from exc
    stream = load_dataset(dataset, name=config, split=split, streaming=True)
    available = set(stream.column_names or [])
    projected = [column for column in SELECTED_COLUMNS if column in available]
    if not projected:
        raise RuntimeError("The remote dataset exposes none of the required columns")
    yield from stream.select_columns(projected).iter(batch_size=batch_size)


def acquire_huggingface(
    output: str | Path,
    *,
    matches: int = 300,
    dataset: str = "gptilt/lol-ultimate-events-challenger-10m",
    split: str = "train_region_americas",
    batch_size: int = 10_000,
    skip_matches: int = 0,
) -> Path:
    """Acquire complete matches and write the canonical raw Parquet file."""
    batches = stream_huggingface_batches(
        dataset,
        split,
        batch_size=batch_size,
    )
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(desc="Streaming events", unit="event")
    except ImportError:
        pass

    def tracked_batches():
        for batch in batches:
            if progress is not None:
                first_column = next(iter(batch.values()), [])
                progress.update(len(first_column))
            yield batch

    try:
        events = collect_complete_matches_from_batches(
            tracked_batches(), matches, skip=skip_matches
        )
    finally:
        if progress is not None:
            progress.close()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(destination, index=False)
    return destination

