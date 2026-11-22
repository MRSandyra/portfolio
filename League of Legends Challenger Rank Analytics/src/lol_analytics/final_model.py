"""Deterministic final-model selection and fitting without holdout scoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .config import features_available_at
from .final_results import authorize_pre_release_tuning
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .modeling import fit_model
from .quality import sha256_file, write_manifest


def select_modal_regularization_c(counts: Mapping[str, int]) -> float:
    """Select the most frequent nested-CV choice; ties use the smaller C."""
    if not counts:
        raise ValueError("selection counts must not be empty")
    parsed: list[tuple[float, int]] = []
    for raw_c, raw_count in counts.items():
        regularization_c = float(raw_c)
        count = int(raw_count)
        if regularization_c <= 0 or count <= 0:
            raise ValueError("selection counts require positive C values and counts")
        parsed.append((regularization_c, count))
    highest_count = max(count for _, count in parsed)
    return min(c_value for c_value, count in parsed if count == highest_count)


def _iso_range(frame: pd.DataFrame) -> dict[str, str]:
    starts = pd.to_datetime(frame["game_start_timestamp_ms"], unit="ms", utc=True)
    return {"start": starts.min().isoformat(), "end": starts.max().isoformat()}


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Final model fitting report",
        "",
        "The final hyperparameters are selected by a frozen deterministic policy: the "
        "modal choice across repeated nested outer folds, with the smaller C used for ties.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Training input: `{manifest['training']['file']['path']}`",
        f"- Selection evidence: `{manifest['selection_evidence']['file']['path']}`",
        "- Frozen future holdout accessed: no",
        "- In-sample performance computed: no",
        "",
        "| Model | Horizon | Eligible training matches | Selected C | Selection counts |",
        "|---|---:|---:|---:|---|",
    ]
    for name, model in manifest["models"].items():
        lines.append(
            f"| {name} | {model['horizon_minutes']} min | "
            f"{model['eligible_training_matches']} | {model['selected_regularization_c']} | "
            f"{model['selection_counts']} |"
        )
    lines.extend(
        [
            "",
            "These fitted artifacts are not performance evidence. Evaluation on the frozen "
            "future holdout requires a separate explicit release step and must preserve the "
            "10-minute early-game versus 20-minute mid-game distinction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_final_models(
    training_input: str | Path,
    selection_evidence_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit one final pipeline per horizon using only frozen development evidence."""
    training_path = Path(training_input)
    evidence_path = Path(selection_evidence_path)
    destination = Path(output_dir)
    manifest_destination = Path(manifest_path)
    if cohort_policy_path is not None:
        authorize_pre_release_tuning(
            cohort_policy_path, training_path, "fit_base_model"
        )
    training_sha256 = sha256_file(training_path)
    evidence_sha256 = sha256_file(evidence_path)

    if manifest_destination.exists():
        existing = json.loads(manifest_destination.read_text(encoding="utf-8"))
        same_inputs = (
            existing.get("training", {}).get("file", {}).get("sha256") == training_sha256
            and existing.get("selection_evidence", {}).get("file", {}).get("sha256")
            == evidence_sha256
            and existing.get("status") == "final_models_fitted_unreleased"
        )
        if not same_inputs:
            raise FileExistsError(
                "Refusing to replace an existing final-model manifest with different inputs"
            )
        for model in existing.get("models", {}).values():
            artifact = Path(model["artifact"]["path"])
            if not artifact.is_file() or sha256_file(artifact) != model["artifact"]["sha256"]:
                raise FileNotFoundError(
                    "Final-model manifest exists but an artifact is missing or changed"
                )
        return existing

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("validation_design") != (
        "repeated_nested_temporal_with_block_bootstrap_refitting"
    ):
        raise ValueError("Selection evidence has the wrong validation design")
    if evidence.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Selection evidence must not access the frozen future holdout")
    evidence_input = Path(str(evidence.get("input", ""))).resolve()
    if evidence_input != training_path.resolve():
        raise ValueError("Selection evidence input does not match the training input")

    events = read_events(training_path)
    match_features = build_match_features(events)
    if int(evidence.get("matches", -1)) != len(match_features):
        raise ValueError("Selection evidence match count does not match training data")

    destination.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    repeats = int(evidence["configuration"]["repeats"])
    outer_splits = int(evidence["configuration"]["outer_splits"])
    expected_decisions = repeats * outer_splits
    for horizon_minutes, key in ((10, "early_10m"), (20, "mid_20m")):
        cohort = eligible_at_horizon(match_features, horizon_minutes)
        features = features_available_at(horizon_minutes * 60_000)
        evidence_model = evidence["models"][key]
        if evidence_model["features"] != features:
            raise ValueError(f"Selection evidence feature contract changed for {key}")
        if int(evidence_model["eligible_matches"]) != len(cohort):
            raise ValueError(f"Selection evidence eligible count changed for {key}")
        if evidence_model["leakage_audit"].get("passed") is not True:
            raise ValueError(f"Selection evidence leakage audit failed for {key}")
        selection_counts = evidence_model["repeated_nested_validation"][
            "selected_c_counts"
        ]
        if sum(int(count) for count in selection_counts.values()) != expected_decisions:
            raise ValueError(f"Selection evidence has incomplete outer-fold decisions for {key}")
        selected_c = select_modal_regularization_c(selection_counts)
        audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
        model = fit_model(
            cohort,
            features,
            regularization_c=selected_c,
        )
        artifact_path = destination / f"final_model_{horizon_minutes}m.joblib"
        joblib.dump(
            {
                "model": model,
                "features": features,
                "horizon_minutes": horizon_minutes,
                "regularization_c": selected_c,
                "fit_cohort": "all_eligible_development_matches",
                "training_sha256": training_sha256,
                "selection_evidence_sha256": evidence_sha256,
            },
            artifact_path,
            compress=3,
        )
        models[key] = {
            "horizon_minutes": horizon_minutes,
            "eligible_training_matches": len(cohort),
            "excluded_before_horizon": len(match_features) - len(cohort),
            "training_time_range_utc": _iso_range(cohort),
            "features": features,
            "leakage_audit": audit.to_dict(),
            "selection_counts": {
                str(c_value): int(count) for c_value, count in selection_counts.items()
            },
            "selection_decisions": expected_decisions,
            "selected_regularization_c": selected_c,
            "artifact": {
                "path": artifact_path.as_posix(),
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "final_models_fitted_unreleased",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_policy": {
            "name": "modal_repeated_nested_outer_fold_choice",
            "rule": "Choose the most frequent C across repeated nested outer folds.",
            "tie_breaker": "Choose the smaller C.",
            "future_holdout_used_for_selection": False,
        },
        "training": {
            "matches": len(match_features),
            "time_range_utc": _iso_range(match_features),
            "file": {
                "path": training_path.as_posix(),
                "bytes": training_path.stat().st_size,
                "sha256": training_sha256,
            },
        },
        "selection_evidence": {
            "validation_design": evidence["validation_design"],
            "repeats": repeats,
            "outer_splits": outer_splits,
            "inner_splits": int(evidence["configuration"]["inner_splits"]),
            "file": {
                "path": evidence_path.as_posix(),
                "bytes": evidence_path.stat().st_size,
                "sha256": evidence_sha256,
            },
        },
        "models": models,
        "evaluation_policy": {
            "in_sample_performance_computed": False,
            "frozen_future_holdout_accessed": False,
            "frozen_future_holdout_scored": False,
            "release_condition": "A separate explicit holdout-release command and authorization.",
        },
    }
    write_manifest(manifest_destination, manifest)
    _write_report(destination / "final_model_report.md", manifest)
    return json.loads(manifest_destination.read_text(encoding="utf-8"))
