from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, TypeVar

from .artifacts import ArtifactValidationError, validate_artifact
from .backtest import (
    BacktestDataError,
    BacktestPolicy,
    build_backtest_report,
    load_json as load_backtest_json,
    load_jsonl as load_backtest_jsonl,
    validate_manifest,
    validate_observations,
)
from .sentiment import ValidationPolicy
from .sentiment_evaluation import (
    EvaluationDataError,
    build_validation_report,
    load_json as load_sentiment_json,
    load_jsonl as load_sentiment_jsonl,
    validate_annotations,
    validate_predictions,
)


class EvidenceVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceVerificationResult:
    report_type: str
    report_id: str
    verified_inputs: tuple[str, ...]
    recomputed_sections: tuple[str, ...]


PolicyT = TypeVar("PolicyT", ValidationPolicy, BacktestPolicy)


def verify_sentiment_evidence(
    report: Mapping[str, Any],
    *,
    annotations_path: Path,
    predictions_path: Path,
    dataset_manifest_path: Path,
    model_manifest_path: Path,
) -> EvidenceVerificationResult:
    _validate_report(report, "sentiment_validation")
    paths = {
        "annotations": annotations_path,
        "predictions": predictions_path,
        "dataset_manifest": dataset_manifest_path,
        "model_manifest": model_manifest_path,
    }
    _verify_bound_inputs(report, paths)
    try:
        annotations = validate_annotations(load_sentiment_jsonl(annotations_path))
        predictions = validate_predictions(load_sentiment_jsonl(predictions_path))
        dataset = load_sentiment_json(dataset_manifest_path)
        model = load_sentiment_json(model_manifest_path)
    except EvaluationDataError as exc:
        raise EvidenceVerificationError(str(exc)) from exc
    annotations_hash = _sha256(annotations_path)
    if dataset.get("content_sha256") != annotations_hash:
        raise EvidenceVerificationError(
            "dataset manifest content_sha256 does not match the annotation file"
        )
    policy = _policy_from_report(report, ValidationPolicy)
    try:
        recomputed = build_validation_report(
            annotation_result=annotations,
            predictions=predictions,
            dataset=dataset,
            model=model,
            report_id=_report_id(report),
            policy=policy,
            evidence=report["evidence"],
        )
    except (EvaluationDataError, TypeError, ValueError) as exc:
        raise EvidenceVerificationError(f"cannot recompute sentiment report: {exc}") from exc
    sections = (
        "dataset",
        "model",
        "agreement",
        "annotation_summary",
        "metrics",
        "slices",
        "evaluation_policy",
    )
    for section in sections:
        _assert_equivalent(report.get(section), recomputed.get(section), section)
    _assert_equivalent(
        _gate_metric_eligibility(report),
        _gate_metric_eligibility(recomputed),
        "activation_gate.metrics_eligible",
    )
    return EvidenceVerificationResult(
        report_type="sentiment_validation",
        report_id=_report_id(report),
        verified_inputs=tuple(paths),
        recomputed_sections=(*sections, "activation_gate.metrics_eligible"),
    )


def verify_backtest_evidence(
    report: Mapping[str, Any],
    *,
    observations_path: Path,
    manifest_path: Path,
) -> EvidenceVerificationResult:
    _validate_report(report, "backtest")
    paths = {"observations": observations_path, "manifest": manifest_path}
    _verify_bound_inputs(report, paths)
    try:
        manifest = validate_manifest(
            load_backtest_json(manifest_path),
            _sha256(observations_path),
        )
        observations = validate_observations(
            load_backtest_jsonl(observations_path),
            manifest,
        )
    except BacktestDataError as exc:
        raise EvidenceVerificationError(str(exc)) from exc
    policy = _policy_from_report(report, BacktestPolicy)
    try:
        recomputed = build_backtest_report(
            split_records=observations,
            manifest=manifest,
            report_id=_report_id(report),
            policy=policy,
            evidence=report["evidence"],
        )
    except (BacktestDataError, TypeError, ValueError) as exc:
        raise EvidenceVerificationError(f"cannot recompute backtest report: {exc}") from exc
    sections = (
        "status",
        "disclaimer",
        "dataset",
        "evaluation_policy",
        "policy_checks",
        "sample",
        "economic_calibration",
        "association",
        "portfolio_simulation",
        "periods",
        "limitations",
    )
    for section in sections:
        _assert_equivalent(report.get(section), recomputed.get(section), section)
    return EvidenceVerificationResult(
        report_type="backtest",
        report_id=_report_id(report),
        verified_inputs=tuple(paths),
        recomputed_sections=sections,
    )


def _validate_report(report: Mapping[str, Any], expected_type: str) -> None:
    try:
        validate_artifact(report, expected_type)
    except ArtifactValidationError as exc:
        raise EvidenceVerificationError(f"report consistency check failed: {exc}") from exc


def _verify_bound_inputs(report: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    evidence = report.get("evidence")
    hashes = evidence.get("input_sha256") if isinstance(evidence, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise EvidenceVerificationError("report does not contain bound input hashes")
    for name, path in paths.items():
        expected = hashes.get(name)
        actual = _sha256(path)
        if expected != actual:
            raise EvidenceVerificationError(f"bound input SHA-256 mismatch for {name}")


def _policy_from_report(report: Mapping[str, Any], policy_type: type[PolicyT]) -> PolicyT:
    value = report.get("evaluation_policy")
    if not isinstance(value, Mapping):
        raise EvidenceVerificationError(
            "report lacks evaluation_policy and cannot be reproduced exactly"
        )
    try:
        policy = policy_type(**dict(value))
    except TypeError as exc:
        raise EvidenceVerificationError(f"invalid evaluation_policy: {exc}") from exc
    if asdict(policy) != dict(value):
        raise EvidenceVerificationError("evaluation_policy contains non-canonical values")
    for name, item in asdict(policy).items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and (not math.isfinite(item) or item < 0):
            raise EvidenceVerificationError(f"evaluation_policy {name} must be non-negative")
        if isinstance(item, str) and not item.strip():
            raise EvidenceVerificationError(f"evaluation_policy {name} cannot be empty")
    return policy


def _gate_metric_eligibility(report: Mapping[str, Any]) -> object:
    gate = report.get("activation_gate")
    return gate.get("metrics_eligible") if isinstance(gate, Mapping) else None


def _report_id(report: Mapping[str, Any]) -> str:
    value = report.get("report_id")
    if not isinstance(value, str) or not value.strip():
        raise EvidenceVerificationError("report_id is missing")
    return value


def _assert_equivalent(actual: object, expected: object, path: str) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise EvidenceVerificationError(f"recomputed section differs: {path}")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15):
            raise EvidenceVerificationError(f"recomputed section differs: {path}")
        return
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise EvidenceVerificationError(f"recomputed section differs: {path} keys")
        for key in actual:
            _assert_equivalent(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise EvidenceVerificationError(f"recomputed section differs: {path} length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equivalent(left, right, f"{path}[{index}]")
        return
    if actual != expected:
        raise EvidenceVerificationError(f"recomputed section differs: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceVerificationError(f"cannot hash input file: {path}") from exc
    return digest.hexdigest()
