"""Real-data temporal benchmark orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import features_available_at
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .modeling import train_and_evaluate
from .reporting import write_benchmark_report, write_json
from .split import chronological_split
from .validation import (
    bootstrap_auc_interval,
    expanding_window_splits,
    paired_auc_delta_interval,
    temporal_cross_validate_with_predictions,
)


def _evaluate_horizon(
    match_features,
    destination: Path,
    *,
    horizon_minutes: int,
    test_fraction: float,
    bootstrap_iterations: int,
    cv_splits: int,
) -> dict[str, Any]:
    cohort = eligible_at_horizon(match_features, horizon_minutes)
    features = features_available_at(horizon_minutes * 60_000)
    audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
    split = chronological_split(cohort, test_fraction=test_fraction)
    result = train_and_evaluate(split.train, split.test, features)
    probability = result.test_predictions["win_probability"].to_numpy()
    truth = result.test_predictions["win"].to_numpy()
    interval = bootstrap_auc_interval(
        truth, probability, iterations=bootstrap_iterations
    )
    temporal_cv, temporal_predictions = temporal_cross_validate_with_predictions(
        cohort,
        features,
        n_splits=cv_splits,
        bootstrap_iterations=bootstrap_iterations,
    )

    model_name = f"model_{horizon_minutes}m.joblib"
    predictions_name = f"test_predictions_{horizon_minutes}m.csv"
    temporal_predictions_name = f"temporal_predictions_{horizon_minutes}m.csv"
    joblib.dump(
        {"model": result.model, "features": features, "horizon_minutes": horizon_minutes},
        destination / model_name,
    )
    result.test_predictions.to_csv(destination / predictions_name, index=False)
    temporal_predictions.to_csv(destination / temporal_predictions_name, index=False)
    return {
        "horizon_minutes": horizon_minutes,
        "eligible_matches": len(cohort),
        "excluded_before_horizon": len(match_features) - len(cohort),
        "train_matches": len(split.train),
        "test_matches": len(split.test),
        "features": features,
        "leakage_audit": audit.to_dict(),
        "split": split.metadata(),
        "holdout_metrics": result.metrics,
        "holdout_roc_auc_interval": interval,
        "temporal_cv": temporal_cv,
        "model_path": model_name,
        "predictions_path": predictions_name,
        "temporal_predictions_path": temporal_predictions_name,
    }


def _paired_temporal_comparison(
    common,
    destination: Path,
    *,
    cv_splits: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    early_features = features_available_at(10 * 60_000)
    mid_features = features_available_at(20 * 60_000)
    predictions = []
    fold_summaries = []
    for temporal_fold in expanding_window_splits(common, n_splits=cv_splits):
        early = train_and_evaluate(
            temporal_fold.train, temporal_fold.test, early_features
        )
        mid = train_and_evaluate(temporal_fold.train, temporal_fold.test, mid_features)
        paired = early.test_predictions[
            ["matchId", "game_start_timestamp_ms", "win"]
        ].copy()
        paired["early_10m_probability"] = early.test_predictions[
            "win_probability"
        ].to_numpy()
        paired["mid_20m_probability"] = mid.test_predictions[
            "win_probability"
        ].to_numpy()
        paired["fold"] = temporal_fold.fold
        predictions.append(paired)
        fold_summaries.append(
            {
                "fold": temporal_fold.fold,
                "train_matches": len(temporal_fold.train),
                "test_matches": len(temporal_fold.test),
                "early_10m_roc_auc": early.metrics["roc_auc"],
                "mid_20m_roc_auc": mid.metrics["roc_auc"],
                "delta_mid_minus_early": mid.metrics["roc_auc"]
                - early.metrics["roc_auc"],
            }
        )
    pooled = pd.concat(predictions, ignore_index=True).sort_values(
        ["game_start_timestamp_ms", "matchId"], kind="stable"
    )
    if pooled["matchId"].duplicated().any():
        raise AssertionError("A match appears in more than one paired temporal fold")
    output_name = "paired_temporal_predictions.csv"
    pooled.to_csv(destination / output_name, index=False)
    truth = pooled["win"].to_numpy()
    early_probability = pooled["early_10m_probability"].to_numpy()
    mid_probability = pooled["mid_20m_probability"].to_numpy()
    early_interval = bootstrap_auc_interval(
        truth, early_probability, iterations=bootstrap_iterations
    )
    mid_interval = bootstrap_auc_interval(
        truth, mid_probability, iterations=bootstrap_iterations
    )
    return {
        "cohort": "matches_eligible_at_20m",
        "n_splits": len(fold_summaries),
        "pooled_test_matches": len(pooled),
        "early_10m_roc_auc_interval": early_interval,
        "mid_20m_roc_auc_interval": mid_interval,
        "delta_interval": paired_auc_delta_interval(
            truth,
            early_probability,
            mid_probability,
            iterations=bootstrap_iterations,
        ),
        "folds": fold_summaries,
        "predictions_path": output_name,
    }


def run_benchmark(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    test_fraction: float = 0.25,
    bootstrap_iterations: int = 2_000,
    cv_splits: int = 3,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events = read_events(source)
    match_features = build_match_features(events)
    match_features.to_parquet(destination / "match_features.parquet", index=False)

    models = {
        "early_10m": _evaluate_horizon(
            match_features,
            destination,
            horizon_minutes=10,
            test_fraction=test_fraction,
            bootstrap_iterations=bootstrap_iterations,
            cv_splits=cv_splits,
        ),
        "mid_20m": _evaluate_horizon(
            match_features,
            destination,
            horizon_minutes=20,
            test_fraction=test_fraction,
            bootstrap_iterations=bootstrap_iterations,
            cv_splits=cv_splits,
        ),
    }

    common = eligible_at_horizon(match_features, 20)
    common_split = chronological_split(common, test_fraction=test_fraction)
    early_features = features_available_at(10 * 60_000)
    mid_features = features_available_at(20 * 60_000)
    early_paired = train_and_evaluate(common_split.train, common_split.test, early_features)
    mid_paired = train_and_evaluate(common_split.train, common_split.test, mid_features)
    paired_truth = early_paired.test_predictions["win"].to_numpy()
    paired = {
        "cohort": "matches_eligible_at_20m",
        "matches": len(common),
        "train_matches": len(common_split.train),
        "test_matches": len(common_split.test),
        "split": common_split.metadata(),
        "early_10m_roc_auc": early_paired.metrics["roc_auc"],
        "mid_20m_roc_auc": mid_paired.metrics["roc_auc"],
        "delta_interval": paired_auc_delta_interval(
            paired_truth,
            early_paired.test_predictions["win_probability"].to_numpy(),
            mid_paired.test_predictions["win_probability"].to_numpy(),
            iterations=bootstrap_iterations,
        ),
    }
    paired_temporal = _paired_temporal_comparison(
        common,
        destination,
        cv_splits=cv_splits,
        bootstrap_iterations=bootstrap_iterations,
    )

    summary: dict[str, Any] = {
        "input": source.as_posix(),
        "matches": len(match_features),
        "test_fraction": test_fraction,
        "models": models,
        "paired_comparison": paired,
        "paired_temporal_comparison": paired_temporal,
    }
    write_json(destination / "benchmark.json", summary)
    write_benchmark_report(destination / "benchmark.md", summary)
    return summary
