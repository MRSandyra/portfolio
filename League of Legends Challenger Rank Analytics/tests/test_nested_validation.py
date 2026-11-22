import pandas as pd
import pytest

from lol_analytics.nested_validation import (
    nested_temporal_validate,
    repeated_nested_temporal_validate,
)


def _nested_frame(matches: int = 48) -> pd.DataFrame:
    base = 1_742_000_000_000
    return pd.DataFrame(
        {
            "matchId": [f"nested-{index:03d}" for index in range(matches)],
            "game_start_timestamp_ms": [
                base + index * 60_000 for index in range(matches)
            ],
            "win": [index % 2 for index in range(matches)],
            "signal": [float(index % 2) + (index % 3) * 0.01 for index in range(matches)],
        }
    )


def test_nested_validation_selects_only_before_outer_test() -> None:
    summary, predictions = nested_temporal_validate(
        _nested_frame(),
        ["signal"],
        c_grid=[0.01, 0.1, 1.0],
        outer_splits=3,
        inner_splits=2,
        bootstrap_iterations=100,
    )

    assert summary["outer_splits"] == 3
    assert summary["inner_splits"] == 2
    assert summary["pooled_outer_test_matches"] == 36
    assert predictions["matchId"].is_unique
    assert set(predictions["selected_regularization_c"]).issubset({0.01, 0.1, 1.0})
    for fold in summary["folds"]:
        assert fold["selection_max_time_utc"] < fold["outer_test_start_utc"]
        assert len(fold["inner_candidates"]) == 3


def test_nested_validation_is_reproducible() -> None:
    kwargs = {
        "c_grid": [0.01, 0.1],
        "outer_splits": 3,
        "inner_splits": 2,
        "bootstrap_iterations": 100,
    }
    first, first_predictions = nested_temporal_validate(
        _nested_frame(), ["signal"], **kwargs
    )
    second, second_predictions = nested_temporal_validate(
        _nested_frame(), ["signal"], **kwargs
    )
    assert first == second
    pd.testing.assert_frame_equal(first_predictions, second_predictions)


def test_nested_validation_rejects_invalid_c_grid() -> None:
    with pytest.raises(ValueError, match="positive"):
        nested_temporal_validate(
            _nested_frame(),
            ["signal"],
            c_grid=[0.0, 0.1],
            bootstrap_iterations=100,
        )


def test_repeated_nested_validation_shifts_boundaries_without_future_leakage() -> None:
    summary, predictions = repeated_nested_temporal_validate(
        _nested_frame(80),
        ["signal"],
        c_grid=[0.01, 0.1],
        outer_splits=3,
        inner_splits=2,
        repeats=3,
        bootstrap_iterations=100,
        refit_bootstrap_iterations=0,
        refit_block_size=4,
    )
    assert summary["repeats"] == 3
    assert summary["boundary_offsets"][1] == 0
    assert not predictions.duplicated(["repeat", "matchId"]).any()
    assert predictions["repeat"].nunique() == 3
    for repeat in summary["repeat_summaries"]:
        for fold in repeat["nested_validation"]["folds"]:
            assert fold["selection_max_time_utc"] < fold["outer_test_start_utc"]
