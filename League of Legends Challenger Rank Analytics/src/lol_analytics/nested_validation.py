"""Nested expanding-window model selection and unbiased outer evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, features_available_at
from .final_results import authorize_pre_release_tuning
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .modeling import train_and_evaluate
from .reporting import write_json, write_nested_validation_report
from .split import DEFAULT_TIME_COLUMN
from .validation import (
    block_bootstrap_refit_auc_interval,
    bootstrap_auc_interval,
    expanding_window_splits,
    paired_auc_delta_interval,
    repeated_boundary_offsets,
)

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _validated_grid(values: Iterable[float]) -> tuple[float, ...]:
    grid = tuple(sorted(set(float(value) for value in values)))
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("c_grid must contain at least one positive value")
    return grid


def _select_regularization_c(
    outer_training: pd.DataFrame,
    features: list[str],
    *,
    c_grid: tuple[float, ...],
    inner_splits: int,
) -> tuple[float, list[dict[str, Any]], int]:
    inner_folds = expanding_window_splits(outer_training, n_splits=inner_splits)
    candidates: list[dict[str, Any]] = []
    for regularization_c in c_grid:
        scores: list[float] = []
        for fold in inner_folds:
            result = train_and_evaluate(
                fold.train,
                fold.test,
                features,
                regularization_c=regularization_c,
            )
            scores.append(result.metrics["roc_auc"])
        candidates.append(
            {
                "regularization_c": regularization_c,
                "inner_roc_auc_mean": float(np.mean(scores)),
                "inner_roc_auc_std": (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                ),
                "inner_fold_roc_auc": scores,
            }
        )
    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate["inner_roc_auc_mean"],
            candidate["regularization_c"],
        ),
    )
    selection_max_timestamp = max(
        int(fold.test[DEFAULT_TIME_COLUMN].max()) for fold in inner_folds
    )
    return float(selected["regularization_c"]), candidates, selection_max_timestamp


def nested_temporal_validate(
    frame: pd.DataFrame,
    features: list[str],
    *,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    outer_splits: int = 3,
    inner_splits: int = 2,
    bootstrap_iterations: int = 2_000,
    outer_boundary_offset: int = 0,
    refit_bootstrap_iterations: int = 0,
    refit_block_size: int = 50,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select hyperparameters in inner past folds and score untouched outer future folds."""
    grid = _validated_grid(c_grid)
    fold_summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for outer_fold in expanding_window_splits(
        frame,
        n_splits=outer_splits,
        boundary_offset=outer_boundary_offset,
    ):
        selected_c, candidates, selection_max_timestamp = _select_regularization_c(
            outer_fold.train,
            features,
            c_grid=grid,
            inner_splits=inner_splits,
        )
        outer_test_start = int(outer_fold.test[DEFAULT_TIME_COLUMN].min())
        if selection_max_timestamp >= outer_test_start:
            raise AssertionError("Inner model selection reaches the outer test period")
        result = train_and_evaluate(
            outer_fold.train,
            outer_fold.test,
            features,
            regularization_c=selected_c,
        )
        fold_predictions = result.test_predictions.copy()
        fold_predictions["outer_fold"] = outer_fold.fold
        fold_predictions["selected_regularization_c"] = selected_c
        predictions.append(fold_predictions)
        fold_summary: dict[str, Any] = {
            "outer_fold": outer_fold.fold,
            "outer_train_matches": len(outer_fold.train),
            "outer_test_matches": len(outer_fold.test),
            "outer_train_end_utc": pd.to_datetime(
                outer_fold.train[DEFAULT_TIME_COLUMN].max(), unit="ms", utc=True
            ).isoformat(),
            "outer_test_start_utc": pd.to_datetime(
                outer_test_start, unit="ms", utc=True
            ).isoformat(),
            "selection_max_time_utc": pd.to_datetime(
                selection_max_timestamp, unit="ms", utc=True
            ).isoformat(),
            "selected_regularization_c": selected_c,
            "inner_candidates": candidates,
            "outer_metrics": result.metrics,
        }
        if refit_bootstrap_iterations:
            fold_summary["block_bootstrap_refit_auc_interval"] = (
                block_bootstrap_refit_auc_interval(
                    outer_fold.train,
                    outer_fold.test,
                    features,
                    regularization_c=selected_c,
                    iterations=refit_bootstrap_iterations,
                    block_size=refit_block_size,
                    random_state=RANDOM_STATE
                    + outer_fold.fold
                    + abs(outer_boundary_offset) * 101,
                )
            )
        fold_summaries.append(fold_summary)

    out_of_fold = pd.concat(predictions, ignore_index=True).sort_values(
        [DEFAULT_TIME_COLUMN, "matchId"], kind="stable"
    )
    if out_of_fold["matchId"].duplicated().any():
        raise AssertionError("A match appears in more than one nested outer test fold")
    interval = bootstrap_auc_interval(
        out_of_fold["win"].to_numpy(),
        out_of_fold["win_probability"].to_numpy(),
        iterations=bootstrap_iterations,
    )
    selections = Counter(
        fold["selected_regularization_c"] for fold in fold_summaries
    )
    return (
        {
            "outer_splits": len(fold_summaries),
            "inner_splits": inner_splits,
            "outer_boundary_offset": outer_boundary_offset,
            "refit_bootstrap_iterations": refit_bootstrap_iterations,
            "refit_block_size": refit_block_size if refit_bootstrap_iterations else None,
            "c_grid": list(grid),
            "pooled_outer_test_matches": len(out_of_fold),
            "pooled_outer_roc_auc_interval": interval,
            "selected_c_counts": {
                str(key): int(value) for key, value in sorted(selections.items())
            },
            "folds": fold_summaries,
        },
        out_of_fold.reset_index(drop=True),
    )


def repeated_nested_temporal_validate(
    frame: pd.DataFrame,
    features: list[str],
    *,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    outer_splits: int = 3,
    inner_splits: int = 2,
    repeats: int = 5,
    max_shift_fraction: float = 0.4,
    bootstrap_iterations: int = 2_000,
    refit_bootstrap_iterations: int = 100,
    refit_block_size: int = 50,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Repeat nested validation over shifted temporal boundaries.

    Expensive block-bootstrap refitting is restricted to the unshifted base repeat.
    Other repeats measure sensitivity to reasonable choices of temporal cut points.
    """
    offsets = repeated_boundary_offsets(
        frame,
        n_splits=outer_splits,
        repeats=repeats,
        max_shift_fraction=max_shift_fraction,
    )
    repeat_summaries: list[dict[str, Any]] = []
    repeat_predictions: list[pd.DataFrame] = []
    selection_counts: Counter[float] = Counter()
    for repeat_index, offset in enumerate(offsets, start=1):
        validation, predictions = nested_temporal_validate(
            frame,
            features,
            c_grid=c_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            bootstrap_iterations=bootstrap_iterations,
            outer_boundary_offset=offset,
            refit_bootstrap_iterations=(
                refit_bootstrap_iterations if offset == 0 else 0
            ),
            refit_block_size=refit_block_size,
        )
        predictions = predictions.copy()
        predictions["repeat"] = repeat_index
        predictions["boundary_offset"] = offset
        repeat_predictions.append(predictions)
        for selected_c, count in validation["selected_c_counts"].items():
            selection_counts[float(selected_c)] += int(count)
        repeat_summaries.append(
            {
                "repeat": repeat_index,
                "boundary_offset": offset,
                "is_base_repeat": offset == 0,
                "nested_validation": validation,
            }
        )

    combined = pd.concat(repeat_predictions, ignore_index=True).sort_values(
        ["repeat", DEFAULT_TIME_COLUMN, "matchId"], kind="stable"
    )
    if combined.duplicated(["repeat", "matchId"]).any():
        raise AssertionError("A match appears twice within one repeated outer test set")
    aucs = np.asarray(
        [
            repeat["nested_validation"]["pooled_outer_roc_auc_interval"]["estimate"]
            for repeat in repeat_summaries
        ],
        dtype=float,
    )
    return (
        {
            "repeats": repeats,
            "boundary_offsets": offsets,
            "max_shift_fraction": max_shift_fraction,
            "outer_splits_per_repeat": outer_splits,
            "inner_splits": inner_splits,
            "c_grid": list(_validated_grid(c_grid)),
            "repeat_roc_auc_mean": float(aucs.mean()),
            "repeat_roc_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "repeat_roc_auc_min": float(aucs.min()),
            "repeat_roc_auc_max": float(aucs.max()),
            "selected_c_counts": {
                str(key): int(value) for key, value in sorted(selection_counts.items())
            },
            "refit_bootstrap": {
                "applied_to_boundary_offset": 0,
                "iterations_per_outer_fold": refit_bootstrap_iterations,
                "block_size_matches": refit_block_size,
            },
            "repeat_summaries": repeat_summaries,
        },
        combined.reset_index(drop=True),
    )


def run_nested_validation(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    outer_splits: int = 3,
    inner_splits: int = 2,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    source = Path(input_path)
    if cohort_policy_path is not None:
        authorize_pre_release_tuning(
            cohort_policy_path, source, "model_selection"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    match_features = build_match_features(read_events(source))

    models: dict[str, Any] = {}
    for horizon_minutes, key in ((10, "early_10m"), (20, "mid_20m")):
        cohort = eligible_at_horizon(match_features, horizon_minutes)
        features = features_available_at(horizon_minutes * 60_000)
        audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
        validation, predictions = nested_temporal_validate(
            cohort,
            features,
            c_grid=c_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            bootstrap_iterations=bootstrap_iterations,
        )
        predictions_name = f"nested_predictions_{horizon_minutes}m.csv"
        predictions.to_csv(destination / predictions_name, index=False)
        models[key] = {
            "horizon_minutes": horizon_minutes,
            "eligible_matches": len(cohort),
            "features": features,
            "leakage_audit": audit.to_dict(),
            "nested_validation": validation,
            "predictions_path": predictions_name,
        }

    common = eligible_at_horizon(match_features, 20)
    paired_results: dict[str, Any] = {}
    paired_predictions: dict[str, pd.DataFrame] = {}
    for horizon_minutes, key in ((10, "early_10m"), (20, "mid_20m")):
        validation, predictions = nested_temporal_validate(
            common,
            features_available_at(horizon_minutes * 60_000),
            c_grid=c_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            bootstrap_iterations=bootstrap_iterations,
        )
        paired_results[key] = validation
        paired_predictions[key] = predictions

    early = paired_predictions["early_10m"]
    mid = paired_predictions["mid_20m"]
    if not early["matchId"].equals(mid["matchId"]):
        raise AssertionError("Paired nested predictions do not align by matchId")
    paired = early[
        ["matchId", DEFAULT_TIME_COLUMN, "win", "outer_fold"]
    ].copy()
    paired["early_10m_probability"] = early["win_probability"].to_numpy()
    paired["mid_20m_probability"] = mid["win_probability"].to_numpy()
    paired["early_selected_c"] = early["selected_regularization_c"].to_numpy()
    paired["mid_selected_c"] = mid["selected_regularization_c"].to_numpy()
    paired_name = "paired_nested_predictions.csv"
    paired.to_csv(destination / paired_name, index=False)
    truth = paired["win"].to_numpy()
    paired_summary = {
        "cohort": "matches_eligible_at_20m",
        "pooled_outer_test_matches": len(paired),
        "early_10m": paired_results["early_10m"],
        "mid_20m": paired_results["mid_20m"],
        "delta_interval": paired_auc_delta_interval(
            truth,
            paired["early_10m_probability"].to_numpy(),
            paired["mid_20m_probability"].to_numpy(),
            iterations=bootstrap_iterations,
        ),
        "predictions_path": paired_name,
    }

    summary: dict[str, Any] = {
        "input": source.as_posix(),
        "matches": len(match_features),
        "validation_design": "nested_expanding_window",
        "models": models,
        "paired_comparison": paired_summary,
    }
    write_json(destination / "nested_validation.json", summary)
    write_nested_validation_report(destination / "nested_validation.md", summary)
    return summary
