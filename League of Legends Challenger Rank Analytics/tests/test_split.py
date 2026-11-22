import pandas as pd
import pytest

from lol_analytics.split import chronological_split

BASE_TIME = 1_742_000_000_000


def test_newest_absolute_times_form_disjoint_holdout() -> None:
    frame = pd.DataFrame(
        {
            "matchId": [f"m{i}" for i in range(10)],
            "match_order": range(10),
            "game_start_timestamp_ms": [BASE_TIME + i * 60_000 for i in range(10)],
            "win": [i % 2 for i in range(10)],
        }
    )
    split = chronological_split(frame, test_fraction=0.2)
    assert split.train["matchId"].tolist() == [f"m{i}" for i in range(8)]
    assert split.test["matchId"].tolist() == ["m8", "m9"]
    assert set(split.train["matchId"]).isdisjoint(split.test["matchId"])
    assert split.boundary_value == BASE_TIME + 8 * 60_000
    assert split.train["game_start_timestamp_ms"].max() < split.test[
        "game_start_timestamp_ms"
    ].min()
    metadata = split.metadata()
    assert metadata["timezone"] == "UTC"
    assert metadata["order_column"] == "game_start_timestamp_ms"
    assert metadata["train_range_utc"]["end"] < metadata["test_range_utc"]["start"]


def test_stream_order_is_not_accepted_as_implicit_chronology() -> None:
    frame = pd.DataFrame(
        {"matchId": ["a", "b", "c", "d"], "match_order": range(4), "win": [0, 1, 0, 1]}
    )
    with pytest.raises(ValueError, match="Stream order is provenance only"):
        chronological_split(frame, test_fraction=0.25)


def test_too_few_matches_before_grouped_boundary_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "matchId": ["a", "b", "c", "d"],
            "game_start_timestamp_ms": [BASE_TIME, BASE_TIME + 1, BASE_TIME + 1, BASE_TIME + 1],
            "win": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="At least two matches must precede"):
        chronological_split(frame, test_fraction=0.25)


def test_matches_sharing_boundary_timestamp_stay_in_test() -> None:
    frame = pd.DataFrame(
        {
            "matchId": ["a", "b", "c", "d", "e", "f"],
            "game_start_timestamp_ms": [
                BASE_TIME,
                BASE_TIME + 1,
                BASE_TIME + 2,
                BASE_TIME + 3,
                BASE_TIME + 3,
                BASE_TIME + 4,
            ],
            "win": [0, 1, 0, 1, 0, 1],
        }
    )
    split = chronological_split(frame, test_fraction=0.34)
    assert split.train["matchId"].tolist() == ["a", "b", "c"]
    assert split.test["matchId"].tolist() == ["d", "e", "f"]
    assert split.train["game_start_timestamp_ms"].max() < split.test[
        "game_start_timestamp_ms"
    ].min()
