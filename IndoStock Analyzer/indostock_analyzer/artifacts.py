from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Mapping

from .output import ReportValidationError, validate_report


FORBIDDEN_DIRS = {
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".uv-cache",
    ".venv", "annotations", "backtests", "build", "data", "datasets", "dist",
    "drivers", "evaluations", "models", "output", "state", "venv",
}
FORBIDDEN_SUFFIXES = {
    ".csv", ".env", ".feather", ".joblib", ".jsonl", ".key", ".onnx", ".parquet",
    ".pem", ".pickle", ".pkl", ".pyc",
}
FORBIDDEN_FILE_ENDINGS = (".checkpoint.json", ".checkpoint.json.lock")
SECRET_PATTERN = re.compile(
    r"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|"
    r"(?i:api[_-]?key|password|client[_-]?secret)\s*[:=]"
)


class ArtifactValidationError(ValueError):
    pass


def load_artifact(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError("artifact must contain a JSON object")
    return value


def validate_artifact(value: Mapping[str, Any], expected_type: str | None = None) -> str:
    artifact_type = value.get("report_type")
    if expected_type and artifact_type != expected_type:
        raise ArtifactValidationError(
            f"expected report_type {expected_type!r}, found {artifact_type!r}"
        )
    if artifact_type == "analysis":
        try:
            validate_report(dict(value))
        except ReportValidationError as exc:
            raise ArtifactValidationError(str(exc)) from exc
    elif artifact_type == "sentiment_validation":
        _require_schema(value, "sentiment_validation")
        _require_mapping(value, "dataset")
        _require_mapping(value, "model")
        _require_mapping(value, "metrics")
        _require_mapping(value, "activation_gate")
        _validate_evidence(value, ("annotations", "predictions", "dataset_manifest", "model_manifest"))
        _validate_sentiment_metrics(value)
    elif artifact_type == "backtest":
        _require_schema(value, "backtest")
        if value.get("status") != "research_only_not_investment_validated":
            raise ArtifactValidationError("backtest status must remain research-only")
        for field in ("dataset", "policy_checks", "economic_calibration", "portfolio_simulation"):
            _require_mapping(value, field)
        if not isinstance(value.get("periods"), list):
            raise ArtifactValidationError("backtest periods must be a list")
        _validate_evidence(value, ("observations", "manifest"))
        _validate_backtest_consistency(value)
    else:
        raise ArtifactValidationError(f"unsupported or missing report_type: {artifact_type!r}")
    return str(artifact_type)


def write_checksum(path: Path) -> Path:
    digest = _sha256(path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    temporary = checksum_path.with_name(f".{checksum_path.name}.tmp")
    try:
        temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
        temporary.replace(checksum_path)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum_path


def verify_checksum(path: Path) -> bool:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    try:
        line = checksum_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read checksum sidecar: {checksum_path}") from exc
    parts = line.split("  ", 1)
    if len(parts) != 2 or parts[1] != path.name or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise ArtifactValidationError("checksum sidecar format is invalid")
    if _sha256(path) != parts[0]:
        raise ArtifactValidationError("artifact checksum mismatch")
    return True


def release_check(root: Path, maximum_bytes: int = 5_000_000) -> list[str]:
    resolved_root = root.resolve()
    findings: list[str] = []
    for path in resolved_root.rglob("*"):
        relative = path.relative_to(resolved_root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            findings.append(f"symbolic link requires review: {relative}")
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRS or path.name == "__pycache__":
                findings.append(f"forbidden directory: {relative}")
            continue
        if any(part in FORBIDDEN_DIRS or part == "__pycache__" for part in relative.parts[:-1]):
            continue
        if (
            path.name == ".env"
            or path.name.startswith(".env.")
            or path.name.endswith(FORBIDDEN_FILE_ENDINGS)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            findings.append(f"forbidden file: {relative}")
            continue
        if path.stat().st_size > maximum_bytes:
            findings.append(f"large file: {relative} ({path.stat().st_size} bytes)")
            continue
        if path.suffix.lower() in {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".in"}:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if SECRET_PATTERN.search(text):
                findings.append(f"potential secret assignment: {relative}")
    return sorted(set(findings))


def _require_schema(value: Mapping[str, Any], report_type: str) -> None:
    if value.get("schema_version") != 1 or value.get("report_type") != report_type:
        raise ArtifactValidationError(f"unsupported {report_type} schema")
    if not isinstance(value.get("report_id"), str) or not value["report_id"].strip():
        raise ArtifactValidationError(f"{report_type} report_id is required")


def _require_mapping(value: Mapping[str, Any], field: str) -> None:
    if not isinstance(value.get(field), Mapping):
        raise ArtifactValidationError(f"artifact field {field} must be an object")


def _validate_evidence(value: Mapping[str, Any], required_inputs: tuple[str, ...]) -> None:
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ArtifactValidationError("artifact field evidence must be an object")
    if evidence.get("level") not in {"generated_from_bound_inputs", "independently_reviewed"}:
        raise ArtifactValidationError("artifact evidence level is unsupported")
    hashes = evidence.get("input_sha256")
    if not isinstance(hashes, Mapping):
        raise ArtifactValidationError("artifact evidence input_sha256 must be an object")
    for name in required_inputs:
        if not isinstance(hashes.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", hashes[name]):
            raise ArtifactValidationError(f"artifact evidence needs a valid SHA-256 for {name}")


def _validate_sentiment_metrics(value: Mapping[str, Any]) -> None:
    metrics = value["metrics"]
    labels = ("NEGATIVE", "NEUTRAL", "POSITIVE")
    confusion = metrics.get("confusion_matrix")
    per_class = metrics.get("per_class")
    if not isinstance(confusion, Mapping) or not isinstance(per_class, Mapping):
        raise ArtifactValidationError("sentiment metrics need confusion_matrix and per_class")
    matrix: dict[str, dict[str, int]] = {}
    for actual in labels:
        row = confusion.get(actual)
        if not isinstance(row, Mapping):
            raise ArtifactValidationError(f"sentiment confusion matrix needs row {actual}")
        matrix[actual] = {}
        for predicted in labels:
            count = row.get(predicted)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ArtifactValidationError("sentiment confusion-matrix counts must be non-negative integers")
            matrix[actual][predicted] = count
    total = sum(sum(row.values()) for row in matrix.values())
    if total <= 0 or metrics.get("evaluated_samples") != total:
        raise ArtifactValidationError("sentiment evaluated_samples is inconsistent with confusion_matrix")
    f1_values: list[float] = []
    for label in labels:
        item = per_class.get(label)
        if not isinstance(item, Mapping):
            raise ArtifactValidationError(f"sentiment per_class needs {label}")
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in labels if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        support = sum(matrix[label].values())
        precision = _divide(true_positive, true_positive + false_positive)
        recall = _divide(true_positive, true_positive + false_negative)
        f1 = _divide(2 * precision * recall, precision + recall)
        expected = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        for field, expected_value in expected.items():
            if not _close(item.get(field), expected_value):
                raise ArtifactValidationError(f"sentiment {label} {field} is inconsistent with confusion_matrix")
        f1_values.append(f1)
    accuracy = sum(matrix[label][label] for label in labels) / total
    if not _close(metrics.get("accuracy"), accuracy):
        raise ArtifactValidationError("sentiment accuracy is inconsistent with confusion_matrix")
    if not _close(metrics.get("macro_f1"), mean(f1_values)):
        raise ArtifactValidationError("sentiment macro_f1 is inconsistent with confusion_matrix")
    ece = _finite(metrics.get("expected_calibration_error"), "sentiment expected_calibration_error")
    if not 0 <= ece <= 1:
        raise ArtifactValidationError("sentiment expected_calibration_error must be within 0..1")
    gate = value["activation_gate"]
    if gate.get("active") is not False:
        raise ArtifactValidationError(
            "stored sentiment evaluation reports are evidence candidates, not runtime activation approvals"
        )
    if not isinstance(gate.get("metrics_eligible"), bool) or not isinstance(gate.get("reasons"), list):
        raise ArtifactValidationError("sentiment activation_gate is malformed")


def _validate_backtest_consistency(value: Mapping[str, Any]) -> None:
    periods = value["periods"]
    if not periods:
        raise ArtifactValidationError("backtest periods cannot be empty")
    sample = value.get("sample")
    portfolio = value["portfolio_simulation"]
    policy = value["policy_checks"]
    if not isinstance(sample, Mapping):
        raise ArtifactValidationError("backtest sample must be an object")
    if sample.get("holdout_periods") != len(periods) or portfolio.get("periods") != len(periods):
        raise ArtifactValidationError("backtest period counts are inconsistent")
    checks = policy.get("checks")
    if not isinstance(checks, Mapping) or not checks or any(not isinstance(item, bool) for item in checks.values()):
        raise ArtifactValidationError("backtest policy checks are malformed")
    if policy.get("passed") is not all(checks.values()):
        raise ArtifactValidationError("backtest policy passed flag is inconsistent")

    net: list[float] = []
    benchmark: list[float] = []
    excess: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    for index, period in enumerate(periods, 1):
        if not isinstance(period, Mapping):
            raise ArtifactValidationError(f"backtest period {index} must be an object")
        period_net = _finite(period.get("net_total_return"), f"backtest period {index} net return")
        period_benchmark = _finite(
            period.get("benchmark_total_return"), f"backtest period {index} benchmark return"
        )
        period_excess = _finite(period.get("net_excess_return"), f"backtest period {index} excess return")
        turnover = _finite(period.get("turnover"), f"backtest period {index} turnover")
        cost = _finite(period.get("cost"), f"backtest period {index} cost")
        if period_net <= -1 or period_benchmark <= -1 or not 0 <= turnover <= 1 or cost < 0:
            raise ArtifactValidationError(f"backtest period {index} contains out-of-bound values")
        if not math.isclose(period_excess, period_net - period_benchmark, rel_tol=1e-9, abs_tol=1e-12):
            raise ArtifactValidationError(f"backtest period {index} excess return is inconsistent")
        net.append(period_net)
        benchmark.append(period_benchmark)
        excess.append(period_excess)
        turnovers.append(turnover)
        costs.append(cost)

    cumulative_net = _compound(net)
    cumulative_benchmark = _compound(benchmark)
    expected = {
        "cumulative_net_total_return": cumulative_net,
        "cumulative_benchmark_total_return": cumulative_benchmark,
        "cumulative_net_excess_return": cumulative_net - cumulative_benchmark,
        "maximum_drawdown": _maximum_drawdown(net),
        "mean_period_excess_return": mean(excess),
        "positive_excess_period_rate": mean(item > 0 for item in excess),
        "mean_turnover": mean(turnovers),
        "total_modeled_cost": sum(costs),
    }
    for field, expected_value in expected.items():
        if not _close(portfolio.get(field), expected_value):
            raise ArtifactValidationError(f"backtest {field} is inconsistent with periods")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{name} must be finite")
    return number


def _close(value: object, expected: float | int) -> bool:
    try:
        actual = _finite(value, "metric")
    except ArtifactValidationError:
        return False
    return math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-12)


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compound(values: list[float]) -> float:
    wealth = 1.0
    for value in values:
        wealth *= 1 + value
    return wealth - 1


def _maximum_drawdown(values: list[float]) -> float:
    wealth = peak = 1.0
    maximum = 0.0
    for value in values:
        wealth *= 1 + value
        peak = max(peak, wealth)
        maximum = min(maximum, wealth / peak - 1)
    return maximum


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read artifact: {path}") from exc
    return digest.hexdigest()
