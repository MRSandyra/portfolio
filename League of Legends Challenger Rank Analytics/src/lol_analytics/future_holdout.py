"""Registration of an unscored, strictly future holdout cohort."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .data import read_events
from .quality import audit_events, sha256_file, write_manifest


def _match_metadata(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("matchId", sort=False)
        .agg(
            game_start_timestamp_ms=("gameStartTimestamp", "first"),
            region=("region", "first"),
        )
        .reset_index()
    )


def build_future_holdout_manifest(
    training_events: pd.DataFrame,
    holdout_events: pd.DataFrame,
    *,
    expected_matches: int,
    allow_region_shift: bool = False,
    prior_evaluated_events: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Validate an outcome-blind holdout contract without fitting or scoring a model."""
    holdout_audit = audit_events(holdout_events, expected_matches=expected_matches)
    if not holdout_audit["passed"]:
        raise ValueError(
            "Future holdout failed data-quality gates: "
            + ", ".join(holdout_audit["issues"])
        )

    training = _match_metadata(training_events)
    holdout = _match_metadata(holdout_events)
    overlap = sorted(set(training["matchId"]).intersection(holdout["matchId"]))
    if overlap:
        raise ValueError(
            "Training and future holdout match IDs must be disjoint; "
            f"found {len(overlap)} overlap(s): {overlap[:5]}"
        )

    prior_cohorts: list[dict[str, Any]] = []
    for name, events in (prior_evaluated_events or {}).items():
        prior_ids = set(events["matchId"].dropna().astype(str))
        prior_overlap = sorted(set(holdout["matchId"].astype(str)).intersection(prior_ids))
        if prior_overlap:
            raise ValueError(
                "Future holdout must be disjoint from every previously evaluated cohort; "
                f"{name} has {len(prior_overlap)} overlap(s): {prior_overlap[:5]}"
            )
        prior_cohorts.append(
            {
                "name": name,
                "matches": int(events["matchId"].nunique()),
                "match_id_overlap": 0,
            }
        )

    training_regions = sorted(training["region"].dropna().astype(str).unique())
    holdout_regions = sorted(holdout["region"].dropna().astype(str).unique())
    same_regions = bool(training_regions) and training_regions == holdout_regions
    if not training_regions or not holdout_regions:
        raise ValueError("Training and future holdout region sets must be non-empty")
    if not same_regions and not allow_region_shift:
        raise ValueError(
            "Future holdout must use the same non-empty region set as training unless "
            "allow_region_shift=True; "
            f"training={training_regions}, holdout={holdout_regions}"
        )

    training_end_ms = int(training["game_start_timestamp_ms"].max())
    holdout_start_ms = int(holdout["game_start_timestamp_ms"].min())
    if holdout_start_ms <= training_end_ms:
        raise ValueError(
            "Future holdout must start strictly after the complete training period; "
            f"training_end_ms={training_end_ms}, holdout_start_ms={holdout_start_ms}"
        )

    # Outcome distribution and any model result are intentionally excluded.
    return {
        "schema_version": 1,
        "status": "frozen_unscored",
        "validation_design": (
            "future_same_region" if same_regions else "future_external_region"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_reference": {
            "matches": int(len(training)),
            "regions": training_regions,
            "time_range_utc": {
                "start": pd.to_datetime(
                    training["game_start_timestamp_ms"].min(), unit="ms", utc=True
                ).isoformat(),
                "end": pd.to_datetime(training_end_ms, unit="ms", utc=True).isoformat(),
            },
        },
        "holdout": {
            "matches": int(len(holdout)),
            "rows": int(len(holdout_events)),
            "regions": holdout_regions,
            "patch_distribution": holdout_audit["patch_distribution"],
            "time_range_utc": holdout_audit["time_range_utc"],
            "quality_gates": holdout_audit["gates"],
            "horizon_coverage": holdout_audit["horizon_coverage"],
            "duplicate_event_ids": holdout_audit["duplicate_event_ids"],
            "critical_column_missing_fraction": holdout_audit[
                "critical_column_missing_fraction"
            ],
        },
        "separation": {
            "match_id_overlap": 0,
            "prior_evaluated_cohorts": prior_cohorts,
            "same_regions": same_regions,
            "region_shift_explicitly_allowed": bool(allow_region_shift),
            "strictly_future": True,
            "gap_ms": holdout_start_ms - training_end_ms,
        },
        "evaluation_policy": {
            "model_fitted": False,
            "predictions_written": False,
            "performance_metrics_computed": False,
            "allowed_before_release": [
                "schema_validation",
                "data_quality_gates",
                "checksum_verification",
                "time_and_identity_separation_checks",
            ],
            "release_condition": (
                "Explicit authorization after the model-selection policy and feature contract "
                "are fixed."
            ),
            "interpretation_after_release": (
                "Combined future-time and external-region validation; region and time effects "
                "cannot be attributed separately."
                if not same_regions
                else "Future-time validation within the training region."
            ),
        },
    }


def freeze_future_holdout(
    training_input: str | Path,
    holdout_input: str | Path,
    manifest_path: str | Path,
    *,
    expected_matches: int,
    allow_region_shift: bool = False,
    source: dict[str, Any] | None = None,
    prior_cohort_inputs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate files and write an idempotent checksum-bound holdout manifest."""
    training_path = Path(training_input)
    holdout_path = Path(holdout_input)
    training_sha256 = sha256_file(training_path)
    holdout_sha256 = sha256_file(holdout_path)
    destination = Path(manifest_path)
    prior_paths = [Path(path) for path in (prior_cohort_inputs or [])]
    prior_checksums = {path.as_posix(): sha256_file(path) for path in prior_paths}

    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        recorded_prior_checksums = {
            cohort["file"]["path"]: cohort["file"]["sha256"]
            for cohort in existing.get("separation", {}).get(
                "prior_evaluated_cohorts", []
            )
            if "file" in cohort
        }
        if (
            existing.get("training_reference", {}).get("file", {}).get("sha256")
            == training_sha256
            and existing.get("holdout", {}).get("file", {}).get("sha256")
            == holdout_sha256
            and existing.get("status") == "frozen_unscored"
            and recorded_prior_checksums == prior_checksums
        ):
            return existing
        raise FileExistsError(
            f"Refusing to replace existing holdout manifest with different inputs: {destination}"
        )

    prior_events = {path.as_posix(): read_events(path) for path in prior_paths}
    manifest = build_future_holdout_manifest(
        read_events(training_path),
        read_events(holdout_path),
        expected_matches=expected_matches,
        allow_region_shift=allow_region_shift,
        prior_evaluated_events=prior_events,
    )
    manifest["training_reference"]["file"] = {
        "path": training_path.as_posix(),
        "bytes": training_path.stat().st_size,
        "sha256": training_sha256,
    }
    manifest["holdout"]["file"] = {
        "path": holdout_path.as_posix(),
        "bytes": holdout_path.stat().st_size,
        "sha256": holdout_sha256,
    }
    if source:
        manifest["holdout"]["source"] = source
    for cohort, path in zip(
        manifest["separation"]["prior_evaluated_cohorts"], prior_paths, strict=True
    ):
        cohort["file"] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": prior_checksums[path.as_posix()],
        }
    write_manifest(destination, manifest)
    return manifest
