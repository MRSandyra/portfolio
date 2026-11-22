import json

import pytest

from indostock_analyzer.artifacts import (
    ArtifactValidationError,
    release_check,
    validate_artifact,
    verify_checksum,
    write_checksum,
)


def test_checksum_detects_artifact_tampering(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")
    write_checksum(path)

    assert verify_checksum(path) is True
    path.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="mismatch"):
        verify_checksum(path)


def test_release_check_flags_generated_sensitive_and_large_files(tmp_path):
    (tmp_path / ".env").write_text("PASS" + "WORD=example\n", encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "large.bin").write_bytes(b"x" * 101)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "revocations.checkpoint.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "local.checkpoint.json").write_text("{}\n", encoding="utf-8")

    findings = release_check(tmp_path, maximum_bytes=100)

    assert any(".env" in item for item in findings)
    assert any("model.onnx" in item for item in findings)
    assert any("large.bin" in item for item in findings)
    assert any("state" in item for item in findings)
    assert any("local.checkpoint.json" in item for item in findings)


def test_artifact_validator_rejects_unrecognized_and_non_research_backtest():
    with pytest.raises(ArtifactValidationError, match="unsupported"):
        validate_artifact({"report_type": "unknown"})
    with pytest.raises(ArtifactValidationError, match="research-only"):
        validate_artifact(
            {
                "schema_version": 1,
                "report_type": "backtest",
                "report_id": "r1",
                "status": "production_ready",
            }
        )


def test_sentiment_artifact_contract_accepts_failed_gate():
    artifact = {
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": "s1",
        "dataset": {"content_sha256": "a" * 64},
        "model": {},
        "metrics": {
            "evaluated_samples": 3,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "expected_calibration_error": 0.1,
            "confusion_matrix": {
                "NEGATIVE": {"NEGATIVE": 1, "NEUTRAL": 0, "POSITIVE": 0},
                "NEUTRAL": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 0},
                "POSITIVE": {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 1},
            },
            "per_class": {
                label: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
                for label in ("NEGATIVE", "NEUTRAL", "POSITIVE")
            },
        },
        "activation_gate": {
            "active": False,
            "metrics_eligible": False,
            "reasons": ["insufficient_test_samples"],
        },
        "evidence": {
            "level": "generated_from_bound_inputs",
            "input_sha256": {
                "annotations": "a" * 64,
                "predictions": "b" * 64,
                "dataset_manifest": "c" * 64,
                "model_manifest": "d" * 64,
            },
        },
    }

    assert validate_artifact(artifact) == "sentiment_validation"


def test_checksum_valid_fake_sentiment_metrics_are_still_rejected(tmp_path):
    artifact = {
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": "forged",
        "dataset": {},
        "model": {},
        "metrics": {
            "evaluated_samples": 3,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "expected_calibration_error": 0.0,
            "confusion_matrix": {
                "NEGATIVE": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 0},
                "NEUTRAL": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 0},
                "POSITIVE": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 0},
            },
            "per_class": {
                label: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
                for label in ("NEGATIVE", "NEUTRAL", "POSITIVE")
            },
        },
        "activation_gate": {"active": False, "metrics_eligible": True, "reasons": []},
        "evidence": {
            "level": "generated_from_bound_inputs",
            "input_sha256": {
                "annotations": "a" * 64,
                "predictions": "b" * 64,
                "dataset_manifest": "c" * 64,
                "model_manifest": "d" * 64,
            },
        },
    }
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    write_checksum(path)

    assert verify_checksum(path) is True
    with pytest.raises(ArtifactValidationError, match="inconsistent"):
        validate_artifact(artifact)


def test_backtest_summary_tampering_is_rejected_even_when_shape_is_valid():
    artifact = {
        "schema_version": 1,
        "report_type": "backtest",
        "report_id": "b1",
        "status": "research_only_not_investment_validated",
        "dataset": {},
        "policy_checks": {"passed": True, "checks": {"minimum_sample": True}},
        "sample": {"holdout_periods": 1},
        "economic_calibration": {},
        "portfolio_simulation": {
            "periods": 1,
            "cumulative_net_total_return": 0.50,
            "cumulative_benchmark_total_return": 0.01,
            "cumulative_net_excess_return": 0.49,
            "maximum_drawdown": 0.0,
            "mean_period_excess_return": 0.01,
            "positive_excess_period_rate": 1.0,
            "mean_turnover": 1.0,
            "total_modeled_cost": 0.001,
        },
        "periods": [{
            "net_total_return": 0.02,
            "benchmark_total_return": 0.01,
            "net_excess_return": 0.01,
            "turnover": 1.0,
            "cost": 0.001,
        }],
        "evidence": {
            "level": "generated_from_bound_inputs",
            "input_sha256": {"observations": "a" * 64, "manifest": "b" * 64},
        },
    }

    with pytest.raises(ArtifactValidationError, match="cumulative_net_total_return"):
        validate_artifact(artifact)
