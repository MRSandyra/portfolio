"""External 10-minute validation across rank and historical-era shifts."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .cohort_policy import authorize_cohort_file_operation
from .config import TEN_MINUTES_MS, features_available_at
from .leakage import assert_leakage_safe
from .metrics import classification_metrics
from .quality import sha256_file, write_manifest
from .validation import bootstrap_auc_interval

DATASET_REF = "bobbyscience/league-of-legends-diamond-ranked-games-10-min"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DATASET_API_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
ARCHIVE_MEMBER = "high_diamond_ranked_10min.csv"
EXPECTED_CSV_SHA256 = "82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6"
EXPECTED_ROWS = 9_879

REQUIRED_COLUMNS = (
    "gameId",
    "blueWins",
    "blueFirstBlood",
    "redFirstBlood",
    "blueKills",
    "redKills",
    "blueDeaths",
    "redDeaths",
    "blueEliteMonsters",
    "redEliteMonsters",
    "blueDragons",
    "redDragons",
    "blueHeralds",
    "redHeralds",
    "blueTowersDestroyed",
    "redTowersDestroyed",
    "blueTotalGold",
    "redTotalGold",
    "blueAvgLevel",
    "redAvgLevel",
    "blueTotalExperience",
    "redTotalExperience",
    "blueTotalMinionsKilled",
    "redTotalMinionsKilled",
    "blueTotalJungleMinionsKilled",
    "redTotalJungleMinionsKilled",
    "blueGoldDiff",
    "redGoldDiff",
    "blueExperienceDiff",
    "redExperienceDiff",
)

UNAVAILABLE_FEATURES = (
    "first_dragon_side_10",
    "first_tower_side_10",
    "first_herald_side_10",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_manifest(csv_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    adapted = adapt_high_diamond_10m(frame)
    return {
        "schema_version": 1,
        "dataset": {
            "ref": DATASET_REF,
            "title": "League of Legends Diamond Ranked Games (10 min)",
            "url": DATASET_URL,
            "api_url": DATASET_API_URL,
            "download_url": DOWNLOAD_URL,
            "license": "CC0: Public Domain",
            "last_updated_utc": "2020-04-13T13:53:02.740000+00:00",
            "rank_scope": "Diamond I to Master solo queue",
            "snapshot_horizon_minutes": 10,
            "match_timestamp_available": False,
            "patch_or_season_available": False,
        },
        "file": {
            "path": csv_path.as_posix(),
            "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "rows": len(frame),
            "columns": len(frame.columns),
        },
        "quality": {
            "adapted_rows": len(adapted),
            "unique_game_ids": bool(not frame["gameId"].duplicated().any()),
            "target_distribution": {
                str(key): int(value)
                for key, value in frame["blueWins"].value_counts().sort_index().items()
            },
            "schema_and_redundancy_checks_passed": True,
        },
    }


def acquire_high_diamond_10m(
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Download the pinned public CSV without committing the raw dataset."""
    output = Path(output_path)
    manifest = Path(manifest_path)
    if output.exists():
        if sha256_file(output) != EXPECTED_CSV_SHA256:
            raise FileExistsError("Refusing to replace an existing CSV with a different hash")
    else:
        request = urllib.request.Request(
            DOWNLOAD_URL,
            headers={"User-Agent": "lol-challenger-analytics/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.namelist()
            if members != [ARCHIVE_MEMBER]:
                raise ValueError(f"Unexpected Kaggle archive members: {members}")
            csv_bytes = archive.read(ARCHIVE_MEMBER)
        if _sha256_bytes(csv_bytes) != EXPECTED_CSV_SHA256:
            raise ValueError("Downloaded CSV checksum differs from the pinned source")
        frame = pd.read_csv(io.BytesIO(csv_bytes))
        adapt_high_diamond_10m(frame)
        if len(frame) != EXPECTED_ROWS:
            raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(csv_bytes)
        temporary.replace(output)

    payload = _source_manifest(output)
    if payload["file"]["sha256"] != EXPECTED_CSV_SHA256:
        raise ValueError("Local CSV checksum differs from the pinned source")
    write_manifest(manifest, payload)
    return payload


def _require_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    if not np.array_equal(left.to_numpy(), right.to_numpy()):
        raise ValueError(message)


def adapt_high_diamond_10m(source: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle blue/red snapshot to the Team 100/200 10-minute contract."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(source.columns))
    if missing:
        raise ValueError(f"External dataset is missing required columns: {missing}")
    frame = source.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        bad = sorted(frame.columns[frame.isna().any()].tolist())
        raise ValueError(f"External dataset has missing or non-numeric values: {bad}")
    if frame["gameId"].duplicated().any():
        raise ValueError("External gameId values must be unique")
    if set(frame["blueWins"].unique()) != {0, 1}:
        raise ValueError("External target must contain exactly classes 0 and 1")
    for column in ("blueFirstBlood", "redFirstBlood"):
        if not frame[column].isin((0, 1)).all():
            raise ValueError(f"{column} must be binary")

    _require_equal(
        frame["blueFirstBlood"] + frame["redFirstBlood"],
        pd.Series(1, index=frame.index),
        "First-blood indicators are not complementary",
    )
    _require_equal(
        frame["blueGoldDiff"],
        frame["blueTotalGold"] - frame["redTotalGold"],
        "blueGoldDiff is inconsistent with team totals",
    )
    _require_equal(
        frame["redGoldDiff"],
        -frame["blueGoldDiff"],
        "Red and blue gold differences are not opposites",
    )
    _require_equal(
        frame["blueExperienceDiff"],
        frame["blueTotalExperience"] - frame["redTotalExperience"],
        "blueExperienceDiff is inconsistent with team totals",
    )
    _require_equal(
        frame["redExperienceDiff"],
        -frame["blueExperienceDiff"],
        "Red and blue experience differences are not opposites",
    )
    _require_equal(frame["blueKills"], frame["redDeaths"], "Blue kills != red deaths")
    _require_equal(frame["redKills"], frame["blueDeaths"], "Red kills != blue deaths")
    _require_equal(
        frame["blueEliteMonsters"],
        frame["blueDragons"] + frame["blueHeralds"],
        "Blue elite-monster aggregate is inconsistent",
    )
    _require_equal(
        frame["redEliteMonsters"],
        frame["redDragons"] + frame["redHeralds"],
        "Red elite-monster aggregate is inconsistent",
    )

    adapted = pd.DataFrame(
        {
            "matchId": "kaggle-high-diamond-" + frame["gameId"].astype("int64").astype(str),
            "win": frame["blueWins"].astype("int8"),
            "gold_diff_10": frame["blueGoldDiff"],
            "xp_diff_10": frame["blueExperienceDiff"],
            "cs_diff_10": (
                frame["blueTotalMinionsKilled"]
                + frame["blueTotalJungleMinionsKilled"]
                - frame["redTotalMinionsKilled"]
                - frame["redTotalJungleMinionsKilled"]
            ),
            "jungle_cs_diff_10": (
                frame["blueTotalJungleMinionsKilled"]
                - frame["redTotalJungleMinionsKilled"]
            ),
            "level_diff_10": 5 * (frame["blueAvgLevel"] - frame["redAvgLevel"]),
            "kill_diff_10": frame["blueKills"] - frame["redKills"],
            "tower_diff_10": (
                frame["blueTowersDestroyed"] - frame["redTowersDestroyed"]
            ),
            "dragon_diff_10": frame["blueDragons"] - frame["redDragons"],
            "herald_diff_10": frame["blueHeralds"] - frame["redHeralds"],
            # Baron Nashor is structurally unavailable before its 20-minute spawn.
            "baron_diff_10": 0,
            "first_blood_side_10": np.where(frame["blueFirstBlood"].eq(1), 1, -1),
            # The source has objective counts but not event order. Counts cannot
            # identify first side when both teams took the same objective type.
            "first_dragon_side_10": np.nan,
            "first_tower_side_10": np.nan,
            "first_herald_side_10": np.nan,
        }
    )
    return adapted.loc[:, ["matchId", "win", *features_available_at(TEN_MINUTES_MS)]]


def _resolve_artifact(path_value: str, manifest_path: Path) -> Path:
    artifact = Path(path_value)
    if artifact.is_absolute():
        return artifact
    cwd_candidate = Path.cwd() / artifact
    if cwd_candidate.exists():
        return cwd_candidate
    return manifest_path.parent.parent.parent / artifact


def load_final_early_bundle(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "final_models_fitted_unreleased":
        raise ValueError("Final-model manifest has an unexpected status")
    evaluation = manifest.get("evaluation_policy", {})
    if evaluation.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Final-model manifest indicates future-holdout access")
    if evaluation.get("frozen_future_holdout_scored") is not False:
        raise ValueError("Final-model manifest indicates future-holdout scoring")
    model_record = manifest.get("models", {}).get("early_10m")
    if not model_record or model_record.get("horizon_minutes") != 10:
        raise ValueError("Final-model manifest lacks the 10-minute model")
    expected_features = features_available_at(TEN_MINUTES_MS)
    if model_record.get("features") != expected_features:
        raise ValueError("Final 10-minute feature contract has changed")
    artifact_path = _resolve_artifact(model_record["artifact"]["path"], manifest_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Final 10-minute model not found: {artifact_path}")
    if sha256_file(artifact_path) != model_record["artifact"]["sha256"]:
        raise ValueError("Final 10-minute model checksum does not match its manifest")
    bundle = joblib.load(artifact_path)
    if bundle.get("horizon_minutes") != 10 or bundle.get("features") != expected_features:
        raise ValueError("Final 10-minute model bundle violates its manifest contract")
    if bundle.get("training_sha256") != manifest["training"]["file"]["sha256"]:
        raise ValueError("Final model training checksum does not match its manifest")
    return manifest, bundle, artifact_path


def _write_report(path: Path, result: dict[str, Any]) -> None:
    metric = result["early_10m"]["metrics"]
    interval = result["early_10m"]["roc_auc_interval"]
    lines = [
        "# External rank and historical-era validation",
        "",
        "This report scores only the frozen 10-minute early-game model. The 20-minute "
        "mid-game model is not compatible with this 10-minute-only source and was not loaded.",
        "",
        f"- External rows: {result['external_source']['rows']:,}",
        "- External rank: Diamond I to Master solo queue",
        "- Source last updated: 2020-04-13; exact match timestamps, patch, and season unavailable",
        f"- Early-game ROC-AUC: {metric['roc_auc']:.4f}",
        f"- Stratified bootstrap 95% interval: [{interval['lower']:.4f}, "
        f"{interval['upper']:.4f}]",
        "- Frozen May 2025 future holdout accessed or scored: no",
        "",
        "Three first-objective-side features are absent from the external schema and are "
        "handled by the medians already learned inside the frozen training pipeline. This "
        "is a combined rank, era, and schema-availability stress test; it cannot isolate "
        "those effects or establish exact season generalization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_rank_era(
    input_path: str | Path,
    source_manifest_path: str | Path,
    final_model_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Score only the frozen early-game model on the pinned historical rank cohort."""
    source = Path(input_path)
    source_manifest_file = Path(source_manifest_path)
    final_manifest_file = Path(final_model_manifest_path)
    destination = Path(output_dir)
    if cohort_policy_path is not None:
        authorize_cohort_file_operation(
            cohort_policy_path, source, "diagnostic_evaluation"
        )
    source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
    source_hash = sha256_file(source)
    if source_hash != source_manifest.get("file", {}).get("sha256"):
        raise ValueError("External CSV checksum does not match its source manifest")
    if source_manifest.get("dataset", {}).get("snapshot_horizon_minutes") != 10:
        raise ValueError("External source is not a 10-minute snapshot")

    frame = pd.read_csv(source)
    external = adapt_high_diamond_10m(frame)
    if len(external) != int(source_manifest["file"]["rows"]):
        raise ValueError("External row count does not match its source manifest")
    audit = assert_leakage_safe(
        external,
        features_available_at(TEN_MINUTES_MS),
        TEN_MINUTES_MS,
    )
    final_manifest, bundle, artifact_path = load_final_early_bundle(final_manifest_file)
    features = bundle["features"]
    probability = bundle["model"].predict_proba(external[features])[:, 1]
    metrics = classification_metrics(external["win"].to_numpy(), probability)
    interval = bootstrap_auc_interval(
        external["win"].to_numpy(),
        probability,
        iterations=bootstrap_iterations,
    )
    predictions = external[["matchId", "win"]].copy()
    predictions["win_probability"] = probability

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validation_design": "external_rank_and_historical_era_10m_only",
        "external_source": {
            "dataset_ref": source_manifest["dataset"]["ref"],
            "rows": len(external),
            "rank_scope": source_manifest["dataset"]["rank_scope"],
            "snapshot_horizon_minutes": 10,
            "last_updated_utc": source_manifest["dataset"]["last_updated_utc"],
            "match_timestamp_available": False,
            "patch_or_season_available": False,
            "file_sha256": source_hash,
        },
        "feature_availability": {
            "contract_features": len(features),
            "exact_or_exactly_derived": 10,
            "structural_constant": ["baron_diff_10"],
            "unavailable_imputed_by_frozen_pipeline": list(UNAVAILABLE_FEATURES),
            "objective_counts_not_used_as_first_side_proxies": True,
        },
        "early_10m": {
            "model_artifact": {
                "path": artifact_path.as_posix(),
                "sha256": sha256_file(artifact_path),
                "selected_regularization_c": bundle["regularization_c"],
            },
            "features": features,
            "leakage_audit": audit.to_dict(),
            "metrics": metrics,
            "roc_auc_interval": interval,
        },
        "evaluation_guards": {
            "model_refit": False,
            "mid_20m_model_loaded": False,
            "mid_20m_model_scored": False,
            "frozen_future_holdout_accessed": False,
            "frozen_future_holdout_scored": False,
            "final_manifest_future_holdout_accessed": final_manifest["evaluation_policy"][
                "frozen_future_holdout_accessed"
            ],
        },
        "interpretation": (
            "Combined rank, historical-era, and schema-availability stress test. Exact "
            "season and patch are unavailable, so effects cannot be isolated."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "external_predictions_10m.csv", index=False)
    write_manifest(destination / "rank_era_validation.json", result)
    _write_report(destination / "rank_era_validation.md", result)
    return result
