from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .sentiment import (
    SENTIMENT_LABELS,
    ValidationPolicy,
    evaluate_activation,
    highest_probability_label,
    normalize_probabilities,
)


ANNOTATION_LABELS = (*SENTIMENT_LABELS, "AMBIGUOUS", "IRRELEVANT")


class EvaluationDataError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationDataError(f"cannot read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationDataError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDataError(f"cannot read JSONL file: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise EvaluationDataError(f"record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def validate_annotations(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        record_id = _required_string(record, "id", index)
        if record_id in seen:
            raise EvaluationDataError(f"duplicate annotation id: {record_id}")
        seen.add(record_id)
        text = _required_string(record, "text", index)
        target = _required_string(record, "target", index).upper()
        annotations = record.get("annotations")
        if not isinstance(annotations, list) or len(annotations) < 2:
            raise EvaluationDataError(f"record {record_id} needs at least two annotations")
        annotators: set[str] = set()
        labels: list[str] = []
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise EvaluationDataError(f"record {record_id} has a malformed annotation")
            annotator = _required_string(annotation, "annotator_id", index)
            if annotator in annotators:
                raise EvaluationDataError(f"record {record_id} repeats annotator {annotator}")
            annotators.add(annotator)
            labels.append(_label(annotation.get("label"), record_id))
        adjudicated = _label(record.get("adjudicated_label"), record_id)
        tags_value = record.get("tags", [])
        if not isinstance(tags_value, list) or any(not isinstance(tag, str) or not tag for tag in tags_value):
            raise EvaluationDataError(f"record {record_id} tags must be non-empty strings")
        normalized.append(
            {
                "id": record_id,
                "text": text,
                "target": target,
                "annotation_labels": labels,
                "adjudicated_label": adjudicated,
                "tags": sorted(set(tags_value)),
            }
        )
    if not normalized:
        raise EvaluationDataError("annotation file is empty")
    agreement = calculate_agreement(item["annotation_labels"] for item in normalized)
    distribution = Counter(item["adjudicated_label"] for item in normalized)
    return {
        "records": normalized,
        "summary": {
            "total_records": len(normalized),
            "evaluable_records": sum(item["adjudicated_label"] in SENTIMENT_LABELS for item in normalized),
            "label_distribution": dict(sorted(distribution.items())),
            "agreement": agreement,
        },
    }


def calculate_agreement(label_groups: Iterable[Iterable[str]]) -> dict[str, float | int | None]:
    total_pairs = 0
    agreeing_pairs = 0
    all_labels: Counter[str] = Counter()
    for group in label_groups:
        labels = list(group)
        counts = Counter(labels)
        pairs = len(labels) * (len(labels) - 1) // 2
        total_pairs += pairs
        agreeing_pairs += sum(count * (count - 1) // 2 for count in counts.values())
        all_labels.update(labels)
    if not total_pairs or not all_labels:
        return {"pair_count": 0, "pairwise_agreement": None, "generalized_kappa": None}
    observed = agreeing_pairs / total_pairs
    label_total = sum(all_labels.values())
    expected = sum((count / label_total) ** 2 for count in all_labels.values())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "pair_count": total_pairs,
        "pairwise_agreement": observed,
        "generalized_kappa": kappa,
    }


def validate_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, 1):
        record_id = _required_string(record, "id", index)
        if record_id in predictions:
            raise EvaluationDataError(f"duplicate prediction id: {record_id}")
        label = _label(record.get("label"), record_id, allowed=SENTIMENT_LABELS)
        probabilities_value = record.get("probabilities")
        if not isinstance(probabilities_value, Mapping):
            raise EvaluationDataError(f"prediction {record_id} needs probabilities")
        try:
            probabilities = normalize_probabilities(probabilities_value)
        except ValueError as exc:
            raise EvaluationDataError(f"prediction {record_id}: {exc}") from exc
        try:
            predicted_from_probabilities = highest_probability_label(probabilities)
        except ValueError as exc:
            raise EvaluationDataError(f"prediction {record_id}: {exc}") from exc
        if label != predicted_from_probabilities:
            raise EvaluationDataError(
                f"prediction {record_id} label does not match its highest probability"
            )
        predictions[record_id] = {"label": label, "probabilities": probabilities}
    return predictions


def calculate_metrics(
    annotations: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [item for item in annotations if item["adjudicated_label"] in SENTIMENT_LABELS]
    missing = [item["id"] for item in eligible if item["id"] not in predictions]
    if missing:
        preview = ", ".join(missing[:5])
        raise EvaluationDataError(f"missing predictions for {len(missing)} record(s): {preview}")
    eligible_ids = {item["id"] for item in eligible}
    unknown = sorted(set(predictions) - eligible_ids)
    if unknown:
        raise EvaluationDataError(f"predictions contain unknown/non-evaluable ids: {', '.join(unknown[:5])}")

    confusion = {actual: {predicted: 0 for predicted in SENTIMENT_LABELS} for actual in SENTIMENT_LABELS}
    confidences: list[tuple[float, bool]] = []
    tagged: dict[str, list[tuple[str, str]]] = defaultdict(list)
    correct = 0
    for item in eligible:
        actual = item["adjudicated_label"]
        prediction = predictions[item["id"]]
        predicted = prediction["label"]
        confusion[actual][predicted] += 1
        is_correct = actual == predicted
        correct += int(is_correct)
        confidences.append((max(prediction["probabilities"].values()), is_correct))
        for tag in item.get("tags", []):
            tagged[tag].append((actual, predicted))

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in SENTIMENT_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in SENTIMENT_LABELS if actual != label)
        false_negative = sum(confusion[label][predicted] for predicted in SENTIMENT_LABELS if predicted != label)
        support = sum(confusion[label].values())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(eligible)
    slices = {
        tag: _slice_metrics(values)
        for tag, values in sorted(tagged.items())
    }
    return {
        "evaluated_samples": total,
        "accuracy": _safe_divide(correct, total),
        "macro_f1": sum(f1_values) / len(f1_values),
        "expected_calibration_error": expected_calibration_error(confidences),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "slices": slices,
    }


def expected_calibration_error(
    values: Iterable[tuple[float, bool]], bins: int = 10
) -> float:
    items = list(values)
    if not items:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in items:
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, correct))
    return sum(
        (len(bucket) / len(items))
        * abs(
            sum(int(correct) for _, correct in bucket) / len(bucket)
            - sum(confidence for confidence, _ in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def build_validation_report(
    annotation_result: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    report_id: str,
    policy: ValidationPolicy = ValidationPolicy(),
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = calculate_metrics(annotation_result["records"], predictions)
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": dict(model),
        "dataset": dict(dataset),
        "agreement": annotation_result["summary"]["agreement"],
        "annotation_summary": {
            key: value
            for key, value in annotation_result["summary"].items()
            if key != "agreement"
        },
        "metrics": {key: value for key, value in metrics.items() if key != "slices"},
        "slices": metrics["slices"],
        "evaluation_policy": asdict(policy),
        "evidence": dict(evidence) if evidence is not None else {
            "level": "generated_from_bound_inputs",
            "independently_reviewed": False,
            "input_sha256": {},
        },
    }
    decision = evaluate_activation(report, policy)
    report["activation_gate"] = {
        "active": decision.active,
        "metrics_eligible": decision.metrics_eligible,
        "reasons": list(decision.reasons),
    }
    return report


def compare_validation_reports(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = list(reports)
    if len(values) < 2:
        raise EvaluationDataError("at least two validation reports are required for comparison")
    expected_signature: tuple[object, ...] | None = None
    rows: list[dict[str, Any]] = []
    seen_models: set[tuple[object, object]] = set()
    for report in values:
        dataset = report.get("dataset")
        model = report.get("model")
        metrics = report.get("metrics")
        if not all(isinstance(item, Mapping) for item in (dataset, model, metrics)):
            raise EvaluationDataError("comparison report is missing dataset, model, or metrics")
        signature = (
            dataset.get("name"),
            dataset.get("version"),
            dataset.get("source_url"),
            dataset.get("content_sha256"),
            dataset.get("domain"),
            dataset.get("split_strategy"),
            metrics.get("evaluated_samples"),
        )
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise EvaluationDataError("reports were not evaluated on the same frozen holdout")
        model_key = (model.get("id"), model.get("revision"))
        if model_key in seen_models:
            raise EvaluationDataError("comparison repeats the same model identity")
        seen_models.add(model_key)
        accuracy = _metric_number(metrics.get("accuracy"), "accuracy")
        macro_f1 = _metric_number(metrics.get("macro_f1"), "macro_f1")
        ece = _metric_number(
            metrics.get("expected_calibration_error"), "expected_calibration_error"
        )
        per_class = metrics.get("per_class")
        if not isinstance(per_class, Mapping):
            raise EvaluationDataError("comparison report is missing per-class metrics")
        recalls = []
        for label in SENTIMENT_LABELS:
            item = per_class.get(label)
            recall = item.get("recall") if isinstance(item, Mapping) else None
            if not isinstance(recall, (int, float)):
                raise EvaluationDataError(f"comparison report is missing {label} recall")
            recalls.append(float(recall))
        rows.append(
            {
                "report_id": report.get("report_id"),
                "model_id": model.get("id"),
                "model_revision": model.get("revision"),
                "evaluated_samples": metrics.get("evaluated_samples"),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "minimum_class_recall": min(recalls),
                "expected_calibration_error": ece,
                "activation_gate": report.get("activation_gate"),
            }
        )
    return sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationDataError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _slice_metrics(values: list[tuple[str, str]]) -> dict[str, float | int]:
    correct = sum(actual == predicted for actual, predicted in values)
    recalls = []
    for label in SENTIMENT_LABELS:
        relevant = [predicted for actual, predicted in values if actual == label]
        if relevant:
            recalls.append(sum(predicted == label for predicted in relevant) / len(relevant))
    return {
        "samples": len(values),
        "accuracy": _safe_divide(correct, len(values)),
        "macro_recall_present_classes": sum(recalls) / len(recalls) if recalls else 0.0,
    }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationDataError(f"comparison report has invalid {name}")
    return float(value)


def _required_string(record: Mapping[str, Any], key: str, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"record {index} needs non-empty {key}")
    return value.strip()


def _label(value: object, record_id: str, allowed: Iterable[str] = ANNOTATION_LABELS) -> str:
    normalized = value.upper() if isinstance(value, str) else ""
    choices = tuple(allowed)
    if normalized not in choices:
        raise EvaluationDataError(f"record {record_id} has invalid label {value!r}")
    return normalized
