import hashlib
import json

import pytest

from indostock_analyzer.sentiment import ValidationPolicy
from indostock_analyzer.artifacts import validate_artifact, verify_checksum
from indostock_analyzer.artifact_cli import main as audit_cli_main
from indostock_analyzer.evidence import EvidenceVerificationError, verify_sentiment_evidence
from indostock_analyzer.sentiment_cli import main as sentiment_cli_main
from indostock_analyzer.sentiment_evaluation import (
    EvaluationDataError,
    build_validation_report,
    calculate_metrics,
    compare_validation_reports,
    validate_annotations,
    validate_predictions,
)


def annotation(record_id, label, tags=None):
    return {
        "id": record_id,
        "text": f"Headline {record_id}",
        "target": "BBRI",
        "annotations": [
            {"annotator_id": "ann-1", "label": label},
            {"annotator_id": "ann-2", "label": label},
        ],
        "adjudicated_label": label,
        "tags": tags or [],
    }


def prediction(record_id, label, confidence=0.8):
    remaining = (1 - confidence) / 2
    probabilities = {name: remaining for name in ("NEGATIVE", "NEUTRAL", "POSITIVE")}
    probabilities[label] = confidence
    return {"id": record_id, "label": label, "probabilities": probabilities}


def test_annotation_validation_detects_duplicates_and_reports_agreement():
    result = validate_annotations(
        [annotation("n1", "NEGATIVE"), annotation("p1", "POSITIVE")]
    )

    assert result["summary"]["evaluable_records"] == 2
    assert result["summary"]["agreement"]["pairwise_agreement"] == 1.0
    with pytest.raises(EvaluationDataError, match="duplicate annotation id"):
        validate_annotations([annotation("same", "NEUTRAL"), annotation("same", "NEUTRAL")])


def test_metrics_include_boundaries_confusion_calibration_and_slices():
    raw_annotations = [
        annotation("n1", "NEGATIVE", ["negation"]),
        annotation("n2", "NEGATIVE", ["numeric_context"]),
        annotation("u1", "NEUTRAL"),
        annotation("u2", "NEUTRAL"),
        annotation("p1", "POSITIVE"),
        annotation("p2", "POSITIVE"),
    ]
    annotations = validate_annotations(raw_annotations)["records"]
    predictions = validate_predictions(
        [
            prediction("n1", "NEGATIVE"),
            prediction("n2", "NEUTRAL"),
            prediction("u1", "NEUTRAL"),
            prediction("u2", "NEUTRAL"),
            prediction("p1", "POSITIVE"),
            prediction("p2", "POSITIVE"),
        ]
    )

    metrics = calculate_metrics(annotations, predictions)

    assert metrics["evaluated_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)
    assert metrics["expected_calibration_error"] == pytest.approx(abs(5 / 6 - 0.8))
    assert metrics["confusion_matrix"]["NEGATIVE"]["NEUTRAL"] == 1
    assert metrics["slices"]["numeric_context"]["accuracy"] == 0.0


def test_missing_prediction_is_rejected():
    annotations = validate_annotations([annotation("p1", "POSITIVE")])["records"]
    with pytest.raises(EvaluationDataError, match="missing predictions"):
        calculate_metrics(annotations, {})


def test_generated_report_exposes_activation_decision():
    annotation_result = validate_annotations(
        [
            annotation("n", "NEGATIVE"),
            annotation("u", "NEUTRAL"),
            annotation("p", "POSITIVE"),
        ]
    )
    predictions = validate_predictions(
        [prediction("n", "NEGATIVE"), prediction("u", "NEUTRAL"), prediction("p", "POSITIVE")]
    )
    report = build_validation_report(
        annotation_result,
        predictions,
        dataset={
            "name": "fixture",
            "version": "1",
            "license": "CC-BY-4.0",
            "source_url": "https://example.test/dataset",
            "content_sha256": "abc123",
            "domain": "indonesian_financial_headlines",
            "split_strategy": "temporal_holdout",
            "target_aware": True,
        },
        model={
            "id": "fixture/model",
            "revision": "abc",
            "license": "Apache-2.0",
            "source_url": "https://example.test/model",
        },
        report_id="fixture-report",
        policy=ValidationPolicy(
            minimum_test_samples=3,
            minimum_macro_f1=0.9,
            minimum_class_recall=0.9,
            maximum_ece=0.25,
        ),
    )

    assert report["activation_gate"]["active"] is False
    assert report["activation_gate"]["metrics_eligible"] is True
    assert "external_independent_review_receipt_required" in report["activation_gate"]["reasons"]
    assert "report_integrity_not_operator_approved" in report["activation_gate"]["reasons"]


def test_model_comparison_requires_the_same_frozen_holdout():
    annotation_result = validate_annotations(
        [annotation("n", "NEGATIVE"), annotation("u", "NEUTRAL"), annotation("p", "POSITIVE")]
    )
    predictions = validate_predictions(
        [prediction("n", "NEGATIVE"), prediction("u", "NEUTRAL"), prediction("p", "POSITIVE")]
    )
    dataset = {
        "name": "fixture",
        "version": "1",
        "license": "CC-BY-4.0",
        "source_url": "https://example.test/dataset",
        "content_sha256": "abc123",
        "domain": "indonesian_financial_headlines",
        "split_strategy": "temporal_holdout",
        "target_aware": True,
    }
    model_a = {
        "id": "baseline",
        "revision": "1",
        "license": "MIT",
        "source_url": "https://example.test/a",
    }
    model_b = {**model_a, "id": "candidate", "source_url": "https://example.test/b"}
    report_a = build_validation_report(
        annotation_result, predictions, dataset, model_a, "baseline-report"
    )
    report_b = build_validation_report(
        annotation_result, predictions, dataset, model_b, "candidate-report"
    )

    comparison = compare_validation_reports([report_a, report_b])

    assert {row["model_id"] for row in comparison} == {"baseline", "candidate"}
    changed_dataset = {**report_b, "dataset": {**dataset, "version": "2"}}
    with pytest.raises(EvaluationDataError, match="same frozen holdout"):
        compare_validation_reports([report_a, changed_dataset])


def test_evaluation_cli_checks_dataset_hash_and_writes_report(tmp_path):
    annotations_path = tmp_path / "annotations.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path = tmp_path / "dataset.json"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "report.json"
    annotations_path.write_text(
        "\n".join(json.dumps(annotation(key, label)) for key, label in (
            ("n", "NEGATIVE"), ("u", "NEUTRAL"), ("p", "POSITIVE")
        )) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(json.dumps(prediction(key, label)) for key, label in (
            ("n", "NEGATIVE"), ("u", "NEUTRAL"), ("p", "POSITIVE")
        )) + "\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    dataset_path.write_text(
        json.dumps({
            "name": "fixture",
            "version": "1",
            "license": "CC-BY-4.0",
            "source_url": "https://example.test/dataset",
            "content_sha256": content_hash,
            "domain": "indonesian_financial_headlines",
            "split_strategy": "temporal_holdout",
            "target_aware": True,
        }),
        encoding="utf-8",
    )
    model_path.write_text(
        json.dumps({
            "id": "fixture/model",
            "revision": "abc",
            "license": "Apache-2.0",
            "source_url": "https://example.test/model",
        }),
        encoding="utf-8",
    )

    exit_code = sentiment_cli_main([
        "evaluate",
        "--annotations", str(annotations_path),
        "--predictions", str(predictions_path),
        "--dataset-manifest", str(dataset_path),
        "--model-manifest", str(model_path),
        "--report-id", "fixture-report",
        "--output", str(output_path),
    ])

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["report_id"] == "fixture-report"
    assert report["metrics"]["evaluated_samples"] == 3
    assert report["evidence"]["level"] == "generated_from_bound_inputs"
    assert set(report["evidence"]["input_sha256"]) == {
        "annotations", "predictions", "dataset_manifest", "model_manifest"
    }
    assert report["activation_gate"]["active"] is False
    assert "insufficient_test_samples" in report["activation_gate"]["reasons"]
    assert validate_artifact(report) == "sentiment_validation"
    assert verify_checksum(output_path) is True
    verification = verify_sentiment_evidence(
        report,
        annotations_path=annotations_path,
        predictions_path=predictions_path,
        dataset_manifest_path=dataset_path,
        model_manifest_path=model_path,
    )
    assert verification.report_id == "fixture-report"
    assert "metrics" in verification.recomputed_sections
    assert audit_cli_main([
        "verify-sentiment-evidence",
        str(output_path),
        "--annotations", str(annotations_path),
        "--predictions", str(predictions_path),
        "--dataset-manifest", str(dataset_path),
        "--model-manifest", str(model_path),
    ]) == 0

    forged = dict(report)
    forged["metrics"] = {
        **report["metrics"],
        "expected_calibration_error": 0.123456,
    }
    assert validate_artifact(forged) == "sentiment_validation"
    with pytest.raises(EvidenceVerificationError, match="expected_calibration_error"):
        verify_sentiment_evidence(
            forged,
            annotations_path=annotations_path,
            predictions_path=predictions_path,
            dataset_manifest_path=dataset_path,
            model_manifest_path=model_path,
        )
    legacy = dict(report)
    legacy.pop("evaluation_policy")
    with pytest.raises(EvidenceVerificationError, match="lacks evaluation_policy"):
        verify_sentiment_evidence(
            legacy,
            annotations_path=annotations_path,
            predictions_path=predictions_path,
            dataset_manifest_path=dataset_path,
            model_manifest_path=model_path,
        )


def test_evidence_verifier_rejects_substituted_prediction_file(tmp_path):
    annotations_path = tmp_path / "annotations.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path = tmp_path / "dataset.json"
    model_path = tmp_path / "model.json"
    annotations_path.write_text(
        "\n".join(json.dumps(annotation(key, label)) for key, label in (
            ("n", "NEGATIVE"), ("u", "NEUTRAL"), ("p", "POSITIVE")
        )) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(json.dumps(prediction(key, label)) for key, label in (
            ("n", "NEGATIVE"), ("u", "NEUTRAL"), ("p", "POSITIVE")
        )) + "\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    dataset_path.write_text(json.dumps({
        "name": "fixture", "version": "1", "license": "CC-BY-4.0",
        "source_url": "https://example.test/dataset", "content_sha256": content_hash,
        "domain": "indonesian_financial_headlines", "split_strategy": "temporal_holdout",
        "target_aware": True,
    }), encoding="utf-8")
    model_path.write_text(json.dumps({
        "id": "fixture/model", "revision": "abc", "license": "Apache-2.0",
        "source_url": "https://example.test/model",
    }), encoding="utf-8")
    annotation_result = validate_annotations([
        annotation("n", "NEGATIVE"), annotation("u", "NEUTRAL"), annotation("p", "POSITIVE")
    ])
    predictions = validate_predictions([
        prediction("n", "NEGATIVE"), prediction("u", "NEUTRAL"), prediction("p", "POSITIVE")
    ])
    report = build_validation_report(
        annotation_result,
        predictions,
        json.loads(dataset_path.read_text(encoding="utf-8")),
        json.loads(model_path.read_text(encoding="utf-8")),
        "bound-report",
        evidence={
            "level": "generated_from_bound_inputs",
            "independently_reviewed": False,
            "input_sha256": {
                "annotations": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
                "predictions": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "dataset_manifest": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "model_manifest": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
        },
    )
    predictions_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceVerificationError, match="predictions"):
        verify_sentiment_evidence(
            report,
            annotations_path=annotations_path,
            predictions_path=predictions_path,
            dataset_manifest_path=dataset_path,
            model_manifest_path=model_path,
        )
