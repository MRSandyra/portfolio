"""Validated inference behavior for CLI and Python callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tokopedia_sentiment.decision import predict_with_thresholds
from tokopedia_sentiment.preprocessing import normalize_indonesian_text


@dataclass(frozen=True)
class Prediction:
    text: str
    normalized_text: str
    label: int
    sentiment: str
    argmax_sentiment: str
    probabilities: dict[str, float]
    confidence: float
    probabilities_calibrated: bool
    threshold_policy_applied: bool

    @property
    def probability_negative(self) -> float:
        return self.probabilities.get("negative", 0.0)

    @property
    def probability_neutral(self) -> float | None:
        return self.probabilities.get("neutral")

    @property
    def probability_positive(self) -> float:
        return self.probabilities.get("positive", 0.0)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["probability_negative"] = self.probability_negative
        result["probability_neutral"] = self.probability_neutral
        result["probability_positive"] = self.probability_positive
        return result


def predict_reviews(
    model: Any,
    reviews: list[str],
    *,
    probabilities_calibrated: bool = False,
    class_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    majority_class: str | None = None,
) -> list[Prediction]:
    """Predict non-empty reviews and return every class probability.

    ``confidence`` is the winning model probability, not a guarantee of correctness.
    """
    if not reviews:
        raise ValueError("at least one review is required")
    normalized = [normalize_indonesian_text(review) for review in reviews]
    empty_positions = [str(index) for index, text in enumerate(normalized) if not text]
    if empty_positions:
        positions = ", ".join(empty_positions)
        raise ValueError(f"review must contain text after normalization (indexes: {positions})")

    probabilities = model.predict_proba(reviews)
    if class_names is None:
        class_names = (
            ["negative", "positive"]
            if probabilities.shape[1] == 2
            else [
                "negative",
                "neutral",
                "positive",
            ]
        )
    if len(class_names) != probabilities.shape[1]:
        raise ValueError("class_names does not match model probability columns")
    threshold_policy_applied = thresholds is not None
    if thresholds is not None:
        if majority_class is None:
            raise ValueError("majority_class is required with thresholds")
        labels = predict_with_thresholds(
            np.asarray(probabilities), class_names, thresholds, majority_class
        )
    else:
        labels = model.predict(reviews)
    results: list[Prediction] = []
    for raw, cleaned, label, probability in zip(
        reviews, normalized, labels, probabilities, strict=True
    ):
        label_id = int(label)
        class_probabilities = {
            name: float(probability[index]) for index, name in enumerate(class_names)
        }
        argmax_sentiment = class_names[int(np.argmax(probability))]
        results.append(
            Prediction(
                text=raw,
                normalized_text=cleaned,
                label=label_id,
                sentiment=class_names[label_id],
                argmax_sentiment=argmax_sentiment,
                probabilities=class_probabilities,
                confidence=class_probabilities[class_names[label_id]],
                probabilities_calibrated=probabilities_calibrated,
                threshold_policy_applied=threshold_policy_applied,
            )
        )
    return results
