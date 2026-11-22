"""Two-stage positive-vs-rest then negative-vs-neutral classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from tokopedia_sentiment.case_study_data import CLASS_NAMES
from tokopedia_sentiment.case_study_modeling import build_case_study_model


class TwoStageSentimentModel:
    """Compose two calibrated binary estimators into three-class probabilities."""

    def __init__(self, positive_vs_other: Any, negative_vs_neutral: Any) -> None:
        self.positive_vs_other = positive_vs_other
        self.negative_vs_neutral = negative_vs_neutral
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, text: list[str]) -> np.ndarray:
        non_positive_probability = self.positive_vs_other.predict_proba(text)[:, 1]
        conditional = self.negative_vs_neutral.predict_proba(text)
        negative = non_positive_probability * conditional[:, 0]
        neutral = non_positive_probability * conditional[:, 1]
        positive = 1.0 - non_positive_probability
        probabilities = np.column_stack([negative, neutral, positive])
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, text: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(text), axis=1)


@dataclass(frozen=True)
class ArchitectureComparison:
    selected_architecture: str
    single_stage: dict[str, Any]
    two_stage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _calibrate_binary(model: Any, text: list[str], labels: np.ndarray) -> Any:
    counts = np.bincount(labels, minlength=2)
    folds = min(3, int(counts.min()))
    if folds < 2:
        raise ValueError("two-stage calibration needs at least two rows per binary class")
    return CalibratedClassifierCV(estimator=model, method="sigmoid", cv=folds).fit(text, labels)


def fit_two_stage_model(
    text: list[str], labels: list[int], *, parameter: float = 1.0
) -> TwoStageSentimentModel:
    y = np.asarray(labels, dtype=int)
    non_positive = (y != 2).astype(int)
    first = _calibrate_binary(
        build_case_study_model("tfidf_logreg_balanced", parameter), text, non_positive
    )

    minority_mask = y != 2
    minority_text = np.asarray(text, dtype=object)[minority_mask].tolist()
    minority_labels = y[minority_mask]
    second = _calibrate_binary(
        build_case_study_model("tfidf_logreg_balanced", parameter),
        minority_text,
        minority_labels,
    )
    return TwoStageSentimentModel(first, second)


def _architecture_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(probability, axis=1)
    recalls = recall_score(actual, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)},
    }


def compare_architectures(
    single_stage_model: Any,
    two_stage_model: TwoStageSentimentModel,
    validation_text: list[str],
    validation_labels: list[int],
) -> ArchitectureComparison:
    actual = np.asarray(validation_labels, dtype=int)
    single = _architecture_metrics(actual, single_stage_model.predict_proba(validation_text))
    two = _architecture_metrics(actual, two_stage_model.predict_proba(validation_text))
    selected = max(
        (("single_stage", single), ("two_stage", two)),
        key=lambda item: (item[1]["macro_f1"], item[1]["balanced_accuracy"]),
    )[0]
    return ArchitectureComparison(selected, single, two)


def fit_architecture(
    architecture: str,
    text: list[str],
    labels: list[int],
    *,
    single_model_name: str,
    parameter: float,
) -> Any:
    if architecture == "two_stage":
        return fit_two_stage_model(text, labels, parameter=parameter)
    if architecture == "single_stage":
        from tokopedia_sentiment.case_study_modeling import fit_calibrated_model

        return fit_calibrated_model(text, labels, model_name=single_model_name, parameter=parameter)
    raise ValueError(f"unknown architecture: {architecture}")
