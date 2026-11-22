"""One-call orchestration for the reproducible case study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .config import features_available_at
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .modeling import train_and_evaluate
from .reporting import write_comparison_report, write_json, write_report
from .split import chronological_split


def run_reproduction(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    horizon_minutes: int = 10,
    test_fraction: float = 0.25,
) -> dict[str, Any]:
    if horizon_minutes not in {10, 20}:
        raise ValueError("horizon_minutes must be 10 or 20")
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    events = read_events(source)
    match_features = build_match_features(events)
    cohort = eligible_at_horizon(match_features, horizon_minutes)
    features = features_available_at(horizon_minutes * 60_000)
    audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
    split = chronological_split(cohort, test_fraction=test_fraction)
    result = train_and_evaluate(split.train, split.test, features)

    match_features.to_parquet(destination / "match_features.parquet", index=False)
    result.test_predictions.to_csv(destination / "test_predictions.csv", index=False)
    joblib.dump(
        {"model": result.model, "features": features, "horizon_minutes": horizon_minutes},
        destination / "model.joblib",
    )

    summary: dict[str, Any] = {
        "input": source.as_posix(),
        "matches": len(match_features),
        "eligible_matches": len(cohort),
        "excluded_before_horizon": len(match_features) - len(cohort),
        "train_matches": len(split.train),
        "test_matches": len(split.test),
        "split": split.metadata(),
        "horizon_minutes": horizon_minutes,
        "features": features,
        "leakage_audit": audit.to_dict(),
        "metrics": result.metrics,
    }
    write_json(destination / "run.json", summary)
    write_report(destination / "report.md", summary)
    return summary


def run_case_study(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    test_fraction: float = 0.25,
) -> dict[str, Any]:
    """Train separately audited 10-minute and 20-minute models on eligible cohorts."""
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    events = read_events(source)
    match_features = build_match_features(events)
    match_features.to_parquet(destination / "match_features.parquet", index=False)

    models: dict[str, Any] = {}
    for name, horizon_minutes in (("early_10m", 10), ("mid_20m", 20)):
        cohort = eligible_at_horizon(match_features, horizon_minutes)
        split = chronological_split(cohort, test_fraction=test_fraction)
        features = features_available_at(horizon_minutes * 60_000)
        audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
        result = train_and_evaluate(split.train, split.test, features)
        model_name = f"model_{horizon_minutes}m.joblib"
        predictions_name = f"test_predictions_{horizon_minutes}m.csv"
        joblib.dump(
            {
                "model": result.model,
                "features": features,
                "horizon_minutes": horizon_minutes,
            },
            destination / model_name,
        )
        result.test_predictions.to_csv(destination / predictions_name, index=False)
        models[name] = {
            "horizon_minutes": horizon_minutes,
            "features": features,
            "leakage_audit": audit.to_dict(),
            "metrics": result.metrics,
            "eligible_matches": len(cohort),
            "excluded_before_horizon": len(match_features) - len(cohort),
            "train_matches": len(split.train),
            "test_matches": len(split.test),
            "split": split.metadata(),
            "model_path": model_name,
            "predictions_path": predictions_name,
        }

    summary: dict[str, Any] = {
        "input": source.as_posix(),
        "matches": len(match_features),
        "models": models,
    }
    write_json(destination / "run.json", summary)
    write_comparison_report(destination / "report.md", summary)
    return summary

