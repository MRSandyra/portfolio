import pandas as pd
import pytest

from lol_analytics.data import (
    canonicalize_events,
    collect_complete_matches,
    collect_complete_matches_from_batches,
)


def _row(
    match_id: str, timestamp: int, event_type: str, game_start: int
) -> dict[str, object]:
    return {
        "matchId": match_id,
        "timestamp": timestamp,
        "gameStartTimestamp": game_start,
        "type": event_type,
        "winningTeam": 100,
    }


def test_interleaved_matches_get_completion_order_and_incomplete_is_dropped() -> None:
    rows = [
        _row("slow", 10, "START", 1_742_000_100_000),
        _row("fast", 10, "START", 1_742_000_000_000),
        _row("incomplete", 10, "START", 1_742_000_200_000),
        _row("fast", 20, "GAME_END", 1_742_000_000_000),
        _row("slow", 30, "GAME_END", 1_742_000_100_000),
    ]
    events = collect_complete_matches(rows, limit=2)
    assert set(events["matchId"]) == {"fast", "slow"}
    assert events.loc[events["matchId"].eq("fast"), "match_order"].unique().tolist() == [0]
    assert events.loc[events["matchId"].eq("slow"), "match_order"].unique().tolist() == [1]


def test_batch_collection_preserves_completion_order() -> None:
    rows = [
        _row("slow", 10, "START", 1_742_000_100_000),
        _row("fast", 10, "START", 1_742_000_000_000),
        _row("fast", 20, "GAME_END", 1_742_000_000_000),
        _row("slow", 30, "GAME_END", 1_742_000_100_000),
    ]
    batches = [
        {column: [row.get(column) for row in rows[:2]] for column in rows[0]},
        {column: [row.get(column) for row in rows[2:]] for column in rows[0]},
    ]
    events = collect_complete_matches_from_batches(batches, limit=2)
    assert events.loc[events["matchId"].eq("fast"), "match_order"].unique().tolist() == [0]
    assert events.loc[events["matchId"].eq("slow"), "match_order"].unique().tolist() == [1]


def test_acquisition_rejects_when_requested_complete_match_count_is_unavailable() -> None:
    rows = [_row("only", 10, "GAME_END", 1_742_000_000_000)]
    with pytest.raises(ValueError, match="Only 1 complete matches"):
        collect_complete_matches(rows, limit=2)


def test_batch_acquisition_rejects_when_complete_match_count_is_unavailable() -> None:
    rows = [_row("only", 10, "GAME_END", 1_742_000_000_000)]
    batch = {column: [row.get(column) for row in rows] for column in rows[0]}
    with pytest.raises(ValueError, match="Only 1 complete matches"):
        collect_complete_matches_from_batches([batch], limit=2)


def test_batch_acquisition_skips_complete_matches_without_resetting_order() -> None:
    rows = [
        _row("skip", 10, "GAME_END", 1_742_000_000_000),
        _row("keep", 20, "GAME_END", 1_743_000_000_000),
    ]
    batch = {column: [row.get(column) for row in rows] for column in rows[0]}
    events = collect_complete_matches_from_batches([batch], limit=1, skip=1)
    assert events["matchId"].unique().tolist() == ["keep"]
    assert events["match_order"].unique().tolist() == [1]


def test_missing_global_start_timestamp_is_rejected() -> None:
    frame = pd.DataFrame(
        [{"matchId": "m1", "timestamp": 0, "type": "GAME_END", "winningTeam": 100}]
    )
    with pytest.raises(ValueError, match="gameStartTimestamp"):
        canonicalize_events(frame)


def test_inconsistent_global_start_timestamp_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            _row("m1", 0, "START", 1_742_000_000_000),
            _row("m1", 10, "GAME_END", 1_742_000_000_001),
        ]
    )
    with pytest.raises(ValueError, match="constant within each match"):
        canonicalize_events(frame)

