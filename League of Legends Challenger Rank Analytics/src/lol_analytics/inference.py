"""Checksum-bound inference bundles with explicit probability transformation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .calibration import apply_platt_calibrator
from .cohort_policy import authorize_cohort_file_operation
from .config import features_available_at
from .monitoring import build_model_input_contracts, validate_model_input_contract
from .quality import sha256_file, write_manifest


def _resolve_artifact(path_value: str, manifest_path: Path) -> Path:
    artifact = Path(path_value)
    if artifact.is_absolute():
        return artifact
    cwd_candidate = Path.cwd() / artifact
    if cwd_candidate.exists():
        return cwd_candidate
    return manifest_path.parent.parent.parent / artifact


def assemble_inference_bundle(
    base_bundle: dict[str, Any],
    contract: dict[str, Any],
    probability_transform: dict[str, Any],
    *,
    calibrator: Any | None = None,
) -> dict[str, Any]:
    """Validate and assemble one serializable inference bundle without fitting."""
    horizon = int(base_bundle.get("horizon_minutes", -1))
    if horizon not in {10, 20}:
        raise ValueError("Base bundle horizon must be 10 or 20 minutes")
    features = features_available_at(horizon * 60_000)
    if base_bundle.get("features") != features:
        raise ValueError("Base bundle feature contract differs from the registry")
    if contract.get("horizon_minutes") != horizon:
        raise ValueError("Runtime contract horizon differs from the base model")
    if contract.get("required_features") != features:
        raise ValueError("Runtime contract features differ from the base model")
    kind = probability_transform.get("kind")
    if kind == "identity_keep_base_probability":
        if calibrator is not None:
            raise ValueError("Identity probability transform must not include a calibrator")
    elif kind == "platt_logistic_on_probability_logit":
        if calibrator is None:
            raise ValueError("Platt probability transform requires a calibrator")
    else:
        raise ValueError(f"Unsupported probability transform: {kind}")
    if "model" not in base_bundle or not hasattr(base_bundle["model"], "predict_proba"):
        raise ValueError("Base bundle lacks a fitted probability model")
    return {
        "schema_version": 1,
        "horizon_minutes": horizon,
        "features": features,
        "model": base_bundle["model"],
        "regularization_c": base_bundle.get("regularization_c"),
        "fit_cohort": base_bundle.get("fit_cohort"),
        "training_sha256": base_bundle.get("training_sha256"),
        "selection_evidence_sha256": base_bundle.get("selection_evidence_sha256"),
        "runtime_contract": contract,
        "probability_transform": probability_transform,
        "calibrator": calibrator,
    }


def predict_inference_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    allowed_all_missing: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate one-row-per-match input and return explicit served probabilities."""
    horizon = int(bundle["horizon_minutes"])
    features = bundle["features"]
    contract = bundle["runtime_contract"]
    if contract.get("required_features") != features:
        raise ValueError("Serialized runtime contract differs from bundle features")
    validate_model_input_contract(
        frame,
        horizon_minutes=horizon,
        allowed_all_missing=allowed_all_missing,
    )
    raw_probability = bundle["model"].predict_proba(frame[features])[:, 1]
    if not np.isfinite(raw_probability).all():
        raise ValueError("Base model produced non-finite probabilities")
    transform = bundle["probability_transform"]["kind"]
    if transform == "identity_keep_base_probability":
        served_probability = raw_probability.copy()
        calibration_applied = False
    elif transform == "platt_logistic_on_probability_logit":
        served_probability = apply_platt_calibrator(
            bundle["calibrator"], raw_probability
        )
        calibration_applied = True
    else:
        raise ValueError(f"Unsupported serialized probability transform: {transform}")
    provenance = [column for column in ("matchId", "win") if column in frame]
    result = frame[provenance].copy()
    result["raw_probability"] = raw_probability
    result["served_probability"] = served_probability
    result["calibration_applied"] = calibration_applied
    result["horizon_minutes"] = horizon
    result["probability_transform"] = transform
    return result


def _load_base_model_bundle(
    final_manifest: dict[str, Any],
    final_manifest_path: Path,
    key: str,
    horizon: int,
) -> tuple[dict[str, Any], Path]:
    record = final_manifest["models"][key]
    expected_features = features_available_at(horizon * 60_000)
    if record.get("horizon_minutes") != horizon or record.get("features") != expected_features:
        raise ValueError(f"Final model contract changed for {key}")
    artifact_path = _resolve_artifact(record["artifact"]["path"], final_manifest_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Final model artifact is missing: {artifact_path}")
    if sha256_file(artifact_path) != record["artifact"]["sha256"]:
        raise ValueError(f"Final model artifact checksum changed for {key}")
    bundle = joblib.load(artifact_path)
    if bundle.get("horizon_minutes") != horizon or bundle.get("features") != expected_features:
        raise ValueError(f"Serialized final model contract changed for {key}")
    if bundle.get("training_sha256") != final_manifest["training"]["file"]["sha256"]:
        raise ValueError(f"Serialized final model training checksum changed for {key}")
    return bundle, artifact_path


def _load_probability_transform(
    calibration_manifest: dict[str, Any],
    calibration_manifest_path: Path,
    key: str,
    horizon: int,
) -> tuple[dict[str, Any], Any | None]:
    record = calibration_manifest["models"][key]
    decision = record.get("decision")
    if decision == "rejected_keep_base_probability":
        if record.get("calibrator_artifact") is not None:
            raise ValueError(f"Rejected calibration unexpectedly has an artifact for {key}")
        return (
            {
                "kind": "identity_keep_base_probability",
                "calibration_applied": False,
                "decision": decision,
                "reason": "Temporal cross-fitted Platt scaling failed the non-worsening rule.",
                "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
            },
            None,
        )
    if decision != "accepted":
        raise ValueError(f"Unknown calibration decision for {key}: {decision}")
    artifact_record = record.get("calibrator_artifact")
    if not artifact_record:
        raise ValueError(f"Accepted calibration lacks an artifact for {key}")
    artifact_path = _resolve_artifact(artifact_record["path"], calibration_manifest_path)
    if sha256_file(artifact_path) != artifact_record["sha256"]:
        raise ValueError(f"Calibrator checksum changed for {key}")
    calibrator_bundle = joblib.load(artifact_path)
    if calibrator_bundle.get("horizon_minutes") != horizon:
        raise ValueError(f"Calibrator horizon changed for {key}")
    return (
        {
            "kind": "platt_logistic_on_probability_logit",
            "calibration_applied": True,
            "decision": decision,
            "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
            "calibrator_artifact_sha256": artifact_record["sha256"],
        },
        calibrator_bundle["calibrator"],
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 inference-bundle report",
        "",
        "Both final models are packaged with their exact runtime contracts. Phase 2 "
        "rejected Platt scaling at both horizons, so served probability equals raw base "
        "probability and `calibration_applied` is false.",
        "",
        "| Bundle | Horizon | Features | Probability transform |",
        "|---|---:|---:|---|",
    ]
    for key in ("early_10m", "mid_20m"):
        record = manifest["bundles"][key]
        lines.append(
            f"| {key} | {record['horizon_minutes']} min | {len(record['features'])} | "
            f"{record['probability_transform']['kind']} |"
        )
    lines.extend(
        [
            "",
            "No model or calibrator was fitted in this phase. External diagnostic cohorts "
            "and the frozen future holdout were not read or scored.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_inference_bundles(
    development_input: str | Path,
    cohort_policy_path: str | Path,
    final_model_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Package final inference artifacts without fitting or holdout access."""
    development = Path(development_input)
    policy_path = Path(cohort_policy_path)
    final_path = Path(final_model_manifest_path)
    calibration_path = Path(calibration_manifest_path)
    contract_file = Path(contract_path)
    destination = Path(output_dir)
    manifest_destination = Path(manifest_path)

    cohort_key = authorize_cohort_file_operation(
        policy_path, development, "fit_base_model"
    )
    final_manifest = json.loads(final_path.read_text(encoding="utf-8"))
    calibration_manifest = json.loads(calibration_path.read_text(encoding="utf-8"))
    contracts = json.loads(contract_file.read_text(encoding="utf-8"))
    if contracts != build_model_input_contracts():
        raise ValueError("Committed runtime contract differs from the feature registry")
    if final_manifest.get("status") != "final_models_fitted_unreleased":
        raise ValueError("Final-model manifest has an unexpected status")
    final_evaluation = final_manifest.get("evaluation_policy", {})
    if final_evaluation.get("frozen_future_holdout_accessed") is not False:
        raise ValueError("Final model evidence accessed the frozen future holdout")
    if final_evaluation.get("frozen_future_holdout_scored") is not False:
        raise ValueError("Final model evidence scored the frozen future holdout")
    if final_manifest["training"]["file"]["sha256"] != sha256_file(development):
        raise ValueError("Development file differs from final-model evidence")
    if calibration_manifest["development"]["file"]["sha256"] != sha256_file(
        development
    ):
        raise ValueError("Calibration evidence belongs to a different development file")
    if calibration_manifest["evidence"]["cohort_policy_sha256"] != sha256_file(
        policy_path
    ):
        raise ValueError("Calibration evidence belongs to a different cohort policy")
    if calibration_manifest["evidence"]["final_model_manifest_sha256"] != sha256_file(
        final_path
    ):
        raise ValueError("Calibration evidence belongs to different final models")
    method_policy = calibration_manifest.get("method_policy", {})
    for flag in (
        "external_diagnostic_labels_used",
        "future_holdout_accessed",
        "future_holdout_labels_read",
        "future_holdout_scored",
    ):
        if method_policy.get(flag) is not False:
            raise ValueError(f"Calibration evidence violates guard: {flag}")

    if manifest_destination.exists():
        existing = json.loads(manifest_destination.read_text(encoding="utf-8"))
        expected_evidence = {
            "cohort_policy_sha256": sha256_file(policy_path),
            "final_model_manifest_sha256": sha256_file(final_path),
            "calibration_manifest_sha256": sha256_file(calibration_path),
            "runtime_contract_sha256": sha256_file(contract_file),
        }
        if existing.get("evidence") != expected_evidence:
            raise FileExistsError(
                "Refusing to reuse inference manifest with changed source evidence"
            )
        for record in existing.get("bundles", {}).values():
            artifact = Path(record["artifact"]["path"])
            if not artifact.is_file() or sha256_file(artifact) != record["artifact"]["sha256"]:
                raise FileNotFoundError("Inference bundle is missing or changed")
        return existing

    destination.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, Any] = {}
    any_calibrated = False
    for horizon, key in ((10, "early_10m"), (20, "mid_20m")):
        base_bundle, base_artifact = _load_base_model_bundle(
            final_manifest, final_path, key, horizon
        )
        transform, calibrator = _load_probability_transform(
            calibration_manifest, calibration_path, key, horizon
        )
        inference_bundle = assemble_inference_bundle(
            base_bundle,
            contracts["models"][key],
            transform,
            calibrator=calibrator,
        )
        artifact_path = destination / f"inference_bundle_{horizon}m.joblib"
        joblib.dump(inference_bundle, artifact_path, compress=3)
        bundles[key] = {
            "horizon_minutes": horizon,
            "features": inference_bundle["features"],
            "probability_transform": transform,
            "source_model_artifact": {
                "path": final_manifest["models"][key]["artifact"]["path"],
                "sha256": sha256_file(base_artifact),
            },
            "artifact": {
                "path": artifact_path.as_posix(),
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
        }
        any_calibrated = any_calibrated or bool(transform["calibration_applied"])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "inference_bundles_calibrated_unreleased"
            if any_calibrated
            else "inference_bundles_base_probability_unreleased"
        ),
        "development_cohort_key": cohort_key,
        "evidence": {
            "cohort_policy_sha256": sha256_file(policy_path),
            "final_model_manifest_sha256": sha256_file(final_path),
            "calibration_manifest_sha256": sha256_file(calibration_path),
            "runtime_contract_sha256": sha256_file(contract_file),
        },
        "bundles": bundles,
        "evaluation_guards": {
            "model_refit": False,
            "calibrator_refit": False,
            "external_diagnostic_cohort_read": False,
            "future_holdout_accessed": False,
            "future_holdout_scored": False,
        },
        "prediction_api": {
            "outputs": [
                "raw_probability",
                "served_probability",
                "calibration_applied",
                "horizon_minutes",
                "probability_transform",
            ],
            "served_probability_policy": (
                "Use the accepted calibration transform when present; otherwise preserve "
                "the base probability exactly."
            ),
        },
    }
    write_manifest(manifest_destination, manifest)
    _write_report(destination / "inference_bundle_report.md", manifest)
    return manifest


def load_inference_bundle(
    artifact_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Inference bundle checksum differs from its manifest")
    bundle = joblib.load(path)
    assemble_inference_bundle(
        bundle,
        bundle["runtime_contract"],
        bundle["probability_transform"],
        calibrator=bundle.get("calibrator"),
    )
    return bundle
