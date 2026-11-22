import numpy as np
import pandas as pd
import pytest

from lol_analytics.validation import (
    block_bootstrap_refit_auc_interval,
    bootstrap_auc_interval,
    expanding_window_splits,
    independent_auc_delta_interval,
    paired_auc_delta_interval,
    repeated_boundary_offsets,
    temporal_cross_validate_with_predictions,
)


def test_expanding_windows_are_strictly_temporal() -> None:
    base = 1_742_000_000_000
    frame = pd.DataFrame(
        {
            "matchId": [f"m{i}" for i in range(12)],
            "game_start_timestamp_ms": [base + i * 60_000 for i in range(12)],
            "win": [i % 2 for i in range(12)],
        }
    )
    folds = expanding_window_splits(frame, n_splits=3)
    assert len(folds) == 3
    assert [len(fold.train) for fold in folds] == [3, 6, 9]
    assert all(
        fold.train["game_start_timestamp_ms"].max()
        < fold.test["game_start_timestamp_ms"].min()
        for fold in folds
    )


def test_shifted_temporal_boundaries_remain_strict() -> None:
    frame = pd.DataFrame(
        {
            "matchId": [f"shift-{i}" for i in range(40)],
            "game_start_timestamp_ms": [
                1_742_000_000_000 + i * 60_000 for i in range(40)
            ],
            "win": [i % 2 for i in range(40)],
        }
    )
    offsets = repeated_boundary_offsets(frame, n_splits=3, repeats=5)
    assert offsets[2] == 0
    assert offsets == sorted(offsets)
    for offset in offsets:
        folds = expanding_window_splits(frame, n_splits=3, boundary_offset=offset)
        assert all(
            fold.train["game_start_timestamp_ms"].max()
            < fold.test["game_start_timestamp_ms"].min()
            for fold in folds
        )


def test_temporal_oof_predictions_are_unique_and_cover_test_blocks() -> None:
    base = 1_742_000_000_000
    frame = pd.DataFrame(
        {
            "matchId": [f"m{i}" for i in range(12)],
            "game_start_timestamp_ms": [base + i * 60_000 for i in range(12)],
            "win": [i % 2 for i in range(12)],
            "signal": [float(i % 2) for i in range(12)],
        }
    )
    summary, predictions = temporal_cross_validate_with_predictions(
        frame,
        ["signal"],
        n_splits=3,
        bootstrap_iterations=100,
    )

    assert summary["pooled_test_matches"] == 9
    assert summary["pooled_roc_auc_interval"]["iterations"] == 100
    assert predictions["matchId"].is_unique
    assert predictions["fold"].value_counts().sort_index().to_dict() == {1: 3, 2: 3, 3: 3}
    assert set(predictions["matchId"]) == {f"m{i}" for i in range(3, 12)}


def test_bootstrap_auc_interval_is_reproducible_and_contains_estimate() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    first = bootstrap_auc_interval(truth, probability, iterations=200)
    second = bootstrap_auc_interval(truth, probability, iterations=200)
    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_paired_delta_uses_same_bootstrap_rows() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    early = np.array([0.2, 0.3, 0.7, 0.4, 0.6, 0.8])
    mid = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    delta = paired_auc_delta_interval(truth, early, mid, iterations=200)
    assert delta["estimate_mid_minus_early"] > 0
    assert delta["bootstrap_probability_mid_gt_early"] > 0.5


def test_independent_delta_resamples_cohorts_separately() -> None:
    first_truth = np.array([0, 0, 1, 1])
    first_probability = np.array([0.1, 0.4, 0.6, 0.9])
    second_truth = np.array([0, 0, 0, 1, 1, 1])
    second_probability = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    delta = independent_auc_delta_interval(
        first_truth,
        first_probability,
        second_truth,
        second_probability,
        iterations=100,
    )
    assert delta["estimate_second_minus_first"] >= 0
    assert delta["iterations"] == 100


def test_bootstrap_rejects_too_few_iterations() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_auc_interval(np.array([0, 1]), np.array([0.1, 0.9]), iterations=99)


def test_block_bootstrap_refits_are_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "matchId": [f"block-{i}" for i in range(40)],
            "game_start_timestamp_ms": [
                1_742_000_000_000 + i * 60_000 for i in range(40)
            ],
            "win": [i % 2 for i in range(40)],
            "signal": [float(i % 2) + (i % 3) * 0.01 for i in range(40)],
        }
    )
    kwargs = {"regularization_c": 0.1, "iterations": 100, "block_size": 4}
    first = block_bootstrap_refit_auc_interval(
        frame.iloc[:30], frame.iloc[30:], ["signal"], **kwargs
    )
    second = block_bootstrap_refit_auc_interval(
        frame.iloc[:30], frame.iloc[30:], ["signal"], **kwargs
    )
    assert first == second
    assert first["bootstrap_refits"] == 100
    assert first["block_size_matches"] == 4
