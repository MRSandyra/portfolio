from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
from importlib.metadata import entry_points
import json
from pathlib import Path
import re
from typing import Any, Protocol

from .models import SentimentResult


SENTIMENT_LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
INDONESIAN_TEXTBLOB_AUDIT = (
    "TextBlob's default PatternAnalyzer uses an English sentiment lexicon. It is not "
    "validated here for Indonesian financial headlines; Indonesian text is therefore "
    "left unassessed instead of being converted to a misleading neutral score."
)


class SentimentConfigurationError(ValueError):
    pass


class SentimentPredictor(Protocol):
    def predict(self, text: str, target: str | None = None) -> Mapping[str, float]: ...


class SentimentAnalyzer(Protocol):
    def analyze(self, text: str, language: str, target: str | None = None) -> SentimentResult: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ValidationPolicy:
    minimum_test_samples: int = 300
    minimum_macro_f1: float = 0.75
    minimum_class_recall: float = 0.65
    maximum_ece: float = 0.15
    required_domain: str = "indonesian_financial_headlines"
    required_split_strategy: str = "temporal_holdout"
    require_target_aware: bool = True
    require_external_review_receipt: bool = True


@dataclass(frozen=True)
class ValidationDecision:
    active: bool
    metrics_eligible: bool
    reasons: tuple[str, ...]
    report_id: str | None


class TextBlobSentimentAnalyzer:
    """Use TextBlob only for explicitly English text."""

    def __init__(self, blob_factory: Callable[[str], object] | None = None):
        self._blob_factory = blob_factory

    def analyze(
        self,
        text: str,
        language: str,
        target: str | None = None,
    ) -> SentimentResult:
        normalized_language = language.lower().split("-")[0]
        if normalized_language != "en":
            return SentimentResult(
                polarity=None,
                subjectivity=None,
                label="UNASSESSED",
                method="textblob_pattern_analyzer",
                language=language,
                supported=False,
                note=INDONESIAN_TEXTBLOB_AUDIT,
                target=target,
                validated=False,
            )

        factory = self._blob_factory
        if factory is None:
            from textblob import TextBlob

            factory = TextBlob
        blob = factory(text)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)
        label = "POSITIVE" if polarity > 0.2 else "NEGATIVE" if polarity < -0.2 else "NEUTRAL"
        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            label=label,
            method="textblob_pattern_analyzer",
            language=language,
            supported=True,
            note="TextBlob result; no domain-specific calibration has been performed.",
            target=target,
            confidence=None,
            validated=False,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "textblob_pattern_analyzer",
            "indonesian_enabled": False,
            "validated": False,
            "note": INDONESIAN_TEXTBLOB_AUDIT,
        }


class ValidatedIndonesianSentimentAnalyzer:
    def __init__(
        self,
        predictor: SentimentPredictor,
        validation_report: Mapping[str, Any],
        provider_name: str,
        policy: ValidationPolicy = ValidationPolicy(),
        report_integrity_approved: bool = False,
        independent_review_approved: bool = False,
        review_approval: Mapping[str, Any] | None = None,
    ):
        self.predictor = predictor
        self.validation_report = dict(validation_report)
        self.provider_name = provider_name
        self.policy = policy
        self.review_approval = dict(review_approval) if review_approval else None
        self.evidence_level = (
            "externally_signed_review"
            if independent_review_approved
            else _evidence_level(self.validation_report)
        )
        self.decision = evaluate_activation(
            self.validation_report,
            policy,
            report_integrity_approved=report_integrity_approved,
            independent_review_approved=independent_review_approved,
        )

    def analyze(
        self,
        text: str,
        language: str,
        target: str | None = None,
    ) -> SentimentResult:
        if language.lower().split("-")[0] != "id":
            return TextBlobSentimentAnalyzer().analyze(text, language, target)
        if not self.decision.active:
            return SentimentResult(
                polarity=None,
                subjectivity=None,
                label="UNASSESSED",
                method=self.provider_name,
                language=language,
                supported=False,
                note="Validation gate rejected the provider: " + ", ".join(self.decision.reasons),
                target=target,
                validated=False,
                validation_report_id=self.decision.report_id,
                evidence_level=self.evidence_level,
            )
        if self.policy.require_target_aware and not target:
            return SentimentResult(
                polarity=None,
                subjectivity=None,
                label="UNASSESSED",
                method=self.provider_name,
                language=language,
                supported=False,
                note="A target ticker is required by the operator-approved entity-aware model.",
                target=target,
                validated=True,
                validation_report_id=self.decision.report_id,
                evidence_level=self.evidence_level,
            )

        try:
            probabilities = normalize_probabilities(self.predictor.predict(text, target))
            label = highest_probability_label(probabilities)
        except Exception as exc:
            return SentimentResult(
                polarity=None,
                subjectivity=None,
                label="UNASSESSED",
                method=self.provider_name,
                language=language,
                supported=False,
                note=f"Sentiment provider failed gracefully ({type(exc).__name__}).",
                target=target,
                validated=True,
                validation_report_id=self.decision.report_id,
                evidence_level=self.evidence_level,
            )
        confidence = probabilities[label]
        polarity = probabilities["POSITIVE"] - probabilities["NEGATIVE"]
        return SentimentResult(
            polarity=polarity,
            subjectivity=None,
            label=label,
            method=self.provider_name,
            language=language,
            supported=True,
            note="Entity-aware Indonesian result accepted by the operator-pinned evidence gate.",
            target=target,
            confidence=confidence,
            validated=True,
            validation_report_id=self.decision.report_id,
            evidence_level=self.evidence_level,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.provider_name,
            "indonesian_enabled": self.decision.active,
            "validated": self.decision.active,
            "metrics_eligible": self.decision.metrics_eligible,
            "evidence_level": self.evidence_level,
            "review_approval": self.review_approval,
            "validation_report_id": self.decision.report_id,
            "activation_reasons": list(self.decision.reasons),
            "policy": {
                "minimum_test_samples": self.policy.minimum_test_samples,
                "minimum_macro_f1": self.policy.minimum_macro_f1,
                "minimum_class_recall": self.policy.minimum_class_recall,
                "maximum_ece": self.policy.maximum_ece,
                "required_domain": self.policy.required_domain,
                "required_split_strategy": self.policy.required_split_strategy,
                "require_target_aware": self.policy.require_target_aware,
                "require_external_review_receipt": self.policy.require_external_review_receipt,
            },
        }


def load_validation_report(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SentimentConfigurationError(f"cannot read sentiment validation report: {path}") from exc
    if expected_sha256 is not None:
        _validate_expected_digest(expected_sha256)
        actual = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual, expected_sha256):
            raise SentimentConfigurationError("sentiment validation report SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SentimentConfigurationError("sentiment validation report is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SentimentConfigurationError("sentiment validation report is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SentimentConfigurationError("sentiment validation report must be an object")
    return value


def verify_validation_report_digest(path: Path, expected_sha256: str) -> str:
    """Bind runtime activation to the exact report bytes approved by the operator."""
    _validate_expected_digest(expected_sha256)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SentimentConfigurationError(f"cannot hash sentiment validation report: {path}") from exc
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise SentimentConfigurationError("sentiment validation report SHA-256 mismatch")
    return actual


def _validate_expected_digest(expected_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise SentimentConfigurationError(
            "sentiment validation SHA-256 must be 64 lowercase hexadecimal characters"
        )


def evaluate_activation(
    report: Mapping[str, Any],
    policy: ValidationPolicy = ValidationPolicy(),
    *,
    report_integrity_approved: bool = False,
    independent_review_approved: bool = False,
) -> ValidationDecision:
    reasons: list[str] = []
    report_id = report.get("report_id") if isinstance(report.get("report_id"), str) else None
    if report.get("schema_version") != 1:
        reasons.append("unsupported_validation_schema")
    if report.get("report_type") != "sentiment_validation":
        reasons.append("wrong_validation_report_type")
    if report_id is None:
        reasons.append("validation_report_id_missing")
    dataset = report.get("dataset")
    metrics = report.get("metrics")
    model = report.get("model")
    if not isinstance(dataset, Mapping):
        reasons.append("dataset_metadata_missing")
        dataset = {}
    if not isinstance(metrics, Mapping):
        reasons.append("evaluation_metrics_missing")
        metrics = {}
    if not isinstance(model, Mapping):
        reasons.append("model_metadata_missing")
        model = {}

    if dataset.get("domain") != policy.required_domain:
        reasons.append("wrong_evaluation_domain")
    if dataset.get("split_strategy") != policy.required_split_strategy:
        reasons.append("non_temporal_holdout")
    if dataset.get("target_aware") is not policy.require_target_aware:
        reasons.append("target_awareness_requirement_not_met")
    if not all(
        _nonempty_string(dataset.get(key))
        for key in ("name", "version", "license", "source_url", "content_sha256")
    ):
        reasons.append("dataset_provenance_incomplete")
    if not _nonempty_string(model.get("id")) or not _nonempty_string(model.get("revision")):
        reasons.append("model_identity_incomplete")
    if not _nonempty_string(model.get("license")) or not _nonempty_string(model.get("source_url")):
        reasons.append("model_provenance_incomplete")

    test_samples = _number(metrics.get("evaluated_samples"))
    if test_samples is None or test_samples < policy.minimum_test_samples:
        reasons.append("insufficient_test_samples")
    macro_f1 = _number(metrics.get("macro_f1"))
    if macro_f1 is None or macro_f1 < policy.minimum_macro_f1:
        reasons.append("macro_f1_below_threshold")
    ece = _number(metrics.get("expected_calibration_error"))
    if ece is None or ece > policy.maximum_ece:
        reasons.append("calibration_error_above_threshold")

    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        reasons.append("per_class_metrics_missing")
    else:
        for label in SENTIMENT_LABELS:
            values = per_class.get(label)
            recall = _number(values.get("recall")) if isinstance(values, Mapping) else None
            if recall is None or recall < policy.minimum_class_recall:
                reasons.append(f"{label.lower()}_recall_below_threshold")

    metric_reasons = tuple(dict.fromkeys(reasons))
    metrics_eligible = not metric_reasons
    if policy.require_external_review_receipt and not independent_review_approved:
        reasons.append("external_independent_review_receipt_required")
    if not report_integrity_approved:
        reasons.append("report_integrity_not_operator_approved")

    return ValidationDecision(
        not reasons,
        metrics_eligible,
        tuple(dict.fromkeys(reasons)),
        report_id,
    )


def load_sentiment_provider(provider_name: str) -> SentimentPredictor:
    matches = entry_points(group="indostock.sentiment").select(name=provider_name)
    selected = list(matches)
    if not selected:
        raise SentimentConfigurationError(
            f"sentiment provider '{provider_name}' is not installed in entry-point group indostock.sentiment"
        )
    if len(selected) > 1:
        raise SentimentConfigurationError(f"multiple sentiment providers are registered as '{provider_name}'")
    try:
        factory = selected[0].load()
        predictor = factory()
    except Exception as exc:
        raise SentimentConfigurationError(
            f"sentiment provider '{provider_name}' could not be initialized ({type(exc).__name__})"
        ) from exc
    if not callable(getattr(predictor, "predict", None)):
        raise SentimentConfigurationError("sentiment provider must expose predict(text, target)")
    return predictor


def normalize_probabilities(value: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label in SENTIMENT_LABELS:
        raw = value.get(label, value.get(label.lower()))
        number = _number(raw)
        if number is None or number < 0 or number > 1:
            raise SentimentConfigurationError(f"invalid probability for label {label}")
        normalized[label] = number
    total = sum(normalized.values())
    if abs(total - 1.0) > 0.02:
        raise SentimentConfigurationError("sentiment probabilities must sum to 1")
    return {label: probability / total for label, probability in normalized.items()}


def highest_probability_label(probabilities: Mapping[str, float]) -> str:
    maximum = max(probabilities[label] for label in SENTIMENT_LABELS)
    winners = [label for label in SENTIMENT_LABELS if probabilities[label] == maximum]
    if len(winners) != 1:
        raise SentimentConfigurationError("sentiment probabilities have no unique maximum")
    return winners[0]


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _evidence_level(report: Mapping[str, Any]) -> str:
    evidence = report.get("evidence")
    level = evidence.get("level") if isinstance(evidence, Mapping) else None
    return level if isinstance(level, str) and level else "unverified"


