import pytest

from indostock_analyzer.sentiment import (
    SentimentConfigurationError,
    TextBlobSentimentAnalyzer,
    ValidatedIndonesianSentimentAnalyzer,
    ValidationPolicy,
    evaluate_activation,
    highest_probability_label,
    normalize_probabilities,
)


def test_indonesian_headline_is_explicitly_unassessed():
    result = TextBlobSentimentAnalyzer().analyze("Laba BBRI tumbuh pada kuartal ini", "id")

    assert result.supported is False
    assert result.polarity is None
    assert result.label == "UNASSESSED"


def valid_report():
    return {
        "schema_version": 1,
        "report_type": "sentiment_validation",
        "report_id": "candidate-20261122",
        "dataset": {
            "domain": "indonesian_financial_headlines",
            "split_strategy": "temporal_holdout",
            "target_aware": True,
            "name": "fixture",
            "version": "1",
            "license": "CC-BY-4.0",
            "source_url": "https://example.test/dataset",
            "content_sha256": "a" * 64,
        },
        "model": {
            "id": "example/model",
            "revision": "sha256:abc",
            "license": "Apache-2.0",
            "source_url": "https://example.test/model",
        },
        "metrics": {
            "evaluated_samples": 300,
            "macro_f1": 0.8,
            "expected_calibration_error": 0.1,
            "per_class": {
                label: {"recall": 0.75}
                for label in ("NEGATIVE", "NEUTRAL", "POSITIVE")
            },
        },
        "evidence": {
            "level": "independently_reviewed",
            "review": {
                "reviewer_id": "independent-reviewer",
                "reviewed_at": "2026-11-22T00:00:00Z",
                "scope": "dataset, predictions, metrics, and failure slices",
                "conflicts_disclosed": True,
            },
        },
    }


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, text, target=None):
        self.calls.append((text, target))
        return {"NEGATIVE": 0.1, "NEUTRAL": 0.2, "POSITIVE": 0.7}


def test_validated_indonesian_provider_is_target_aware():
    predictor = FakePredictor()
    analyzer = ValidatedIndonesianSentimentAnalyzer(
        predictor,
        valid_report(),
        "fake-provider",
        report_integrity_approved=True,
        independent_review_approved=True,
    )

    result = analyzer.analyze("BBRI membukukan kenaikan laba", "id", target="BBRI")

    assert result.supported is True
    assert result.validated is True
    assert result.label == "POSITIVE"
    assert result.polarity == pytest.approx(0.6)
    assert result.confidence == pytest.approx(0.7)
    assert predictor.calls == [("BBRI membukukan kenaikan laba", "BBRI")]


def test_rejected_report_does_not_invoke_predictor():
    predictor = FakePredictor()
    report = valid_report()
    report["metrics"]["macro_f1"] = 0.4
    analyzer = ValidatedIndonesianSentimentAnalyzer(
        predictor,
        report,
        "fake-provider",
        report_integrity_approved=True,
        independent_review_approved=True,
    )

    result = analyzer.analyze("BBRI membukukan kenaikan laba", "id", target="BBRI")

    assert result.supported is False
    assert result.label == "UNASSESSED"
    assert "macro_f1_below_threshold" in result.note
    assert predictor.calls == []


def test_provider_failure_is_reported_without_breaking_pipeline():
    class FailedPredictor:
        def predict(self, text, target=None):
            raise RuntimeError("model runtime unavailable")

    analyzer = ValidatedIndonesianSentimentAnalyzer(
        FailedPredictor(),
        valid_report(),
        "failed-provider",
        report_integrity_approved=True,
        independent_review_approved=True,
    )

    result = analyzer.analyze("BBRI membukukan kenaikan laba", "id", target="BBRI")

    assert result.supported is False
    assert result.label == "UNASSESSED"
    assert "failed gracefully (RuntimeError)" in result.note


def test_activation_gate_rejects_wrong_domain_and_missing_provenance():
    report = valid_report()
    report["dataset"] = {
        "domain": "social_media",
        "split_strategy": "random",
        "target_aware": False,
    }

    decision = evaluate_activation(report, ValidationPolicy())

    assert decision.active is False
    assert "wrong_evaluation_domain" in decision.reasons
    assert "dataset_provenance_incomplete" in decision.reasons


def test_high_metrics_do_not_activate_without_review_and_operator_digest_approval():
    predictor = FakePredictor()
    report = valid_report()
    report["evidence"] = {
        "level": "generated_from_bound_inputs",
        "independently_reviewed": False,
    }
    analyzer = ValidatedIndonesianSentimentAnalyzer(
        predictor, report, "fake-provider"
    )

    result = analyzer.analyze("BBRI membukukan kenaikan laba", "id", target="BBRI")

    assert analyzer.decision.metrics_eligible is True
    assert analyzer.decision.active is False
    assert "external_independent_review_receipt_required" in analyzer.decision.reasons
    assert "report_integrity_not_operator_approved" in analyzer.decision.reasons
    assert result.supported is False
    assert predictor.calls == []


def test_review_claim_inside_report_without_signed_receipt_stays_inactive():
    decision = evaluate_activation(valid_report())

    assert decision.metrics_eligible is True
    assert decision.active is False
    assert decision.reasons == (
        "external_independent_review_receipt_required",
        "report_integrity_not_operator_approved",
    )


def test_probability_boundaries_are_enforced():
    assert normalize_probabilities(
        {"NEGATIVE": 0.0, "NEUTRAL": 0.2, "POSITIVE": 0.8}
    )["POSITIVE"] == pytest.approx(0.8)
    with pytest.raises(SentimentConfigurationError):
        normalize_probabilities({"NEGATIVE": -0.1, "NEUTRAL": 0.2, "POSITIVE": 0.9})
    with pytest.raises(SentimentConfigurationError):
        normalize_probabilities({"NEGATIVE": 0.1, "NEUTRAL": 0.1, "POSITIVE": 0.1})
    with pytest.raises(SentimentConfigurationError, match="unique maximum"):
        highest_probability_label({"NEGATIVE": 0.4, "NEUTRAL": 0.2, "POSITIVE": 0.4})
