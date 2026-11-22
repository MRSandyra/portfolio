"""Precision-constrained review triage with explicit abstention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class ClassQueuePolicy:
    threshold: float
    minimum_precision: float
    validation_precision: float
    validation_recall: float
    validation_queue_fraction: float
    target_met: bool


@dataclass(frozen=True)
class TriagePolicy:
    queues: dict[str, ClassQueuePolicy]
    positive_threshold: float
    positive_validation_precision: float
    positive_validation_coverage: float
    max_queue_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_queue_threshold(
    binary_actual: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_precision: float,
    max_queue_fraction: float,
) -> ClassQueuePolicy:
    precision, recall, thresholds = precision_recall_curve(binary_actual, probability)
    candidates: list[tuple[float, float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        queue_fraction = float(np.mean(probability >= threshold))
        candidate_precision = float(precision[index])
        candidate_recall = float(recall[index])
        if candidate_precision >= minimum_precision and queue_fraction <= max_queue_fraction:
            candidates.append(
                (candidate_recall, candidate_precision, -queue_fraction, float(threshold))
            )
    if not candidates:
        return ClassQueuePolicy(
            threshold=1.01,
            minimum_precision=minimum_precision,
            validation_precision=0.0,
            validation_recall=0.0,
            validation_queue_fraction=0.0,
            target_met=False,
        )
    recall_value, precision_value, negative_fraction, threshold_value = max(candidates)
    return ClassQueuePolicy(
        threshold=threshold_value,
        minimum_precision=minimum_precision,
        validation_precision=precision_value,
        validation_recall=recall_value,
        validation_queue_fraction=-negative_fraction,
        target_met=True,
    )


def _select_positive_threshold(
    binary_actual: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_precision: float,
) -> tuple[float, float, float]:
    precision, _, thresholds = precision_recall_curve(binary_actual, probability)
    candidates = []
    for index, threshold in enumerate(thresholds):
        coverage = float(np.mean(probability >= threshold))
        if precision[index] >= minimum_precision:
            candidates.append((coverage, float(precision[index]), float(threshold)))
    if not candidates:
        return 1.01, 0.0, 0.0
    coverage, precision_value, threshold = max(candidates)
    return threshold, precision_value, coverage


def select_triage_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    *,
    minimum_precision: dict[str, float] | None = None,
    max_queue_fraction: float = 0.10,
    positive_minimum_precision: float = 0.99,
) -> TriagePolicy:
    """Select queue thresholds on validation only; disable infeasible queues."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    precision_targets = minimum_precision or {"negative": 0.30, "neutral": 0.15}
    queues = {
        name: _select_queue_threshold(
            (actual == class_names.index(name)).astype(int),
            probability[:, class_names.index(name)],
            minimum_precision=target,
            max_queue_fraction=max_queue_fraction,
        )
        for name, target in precision_targets.items()
    }
    positive_id = class_names.index("positive")
    positive_threshold, positive_precision, positive_coverage = _select_positive_threshold(
        (actual == positive_id).astype(int),
        probability[:, positive_id],
        minimum_precision=positive_minimum_precision,
    )
    return TriagePolicy(
        queues=queues,
        positive_threshold=positive_threshold,
        positive_validation_precision=positive_precision,
        positive_validation_coverage=positive_coverage,
        max_queue_fraction=max_queue_fraction,
    )


def apply_triage_policy(
    probabilities: np.ndarray, class_names: list[str], policy: TriagePolicy | dict[str, Any]
) -> np.ndarray:
    """Return negative_candidate, neutral_candidate, positive, or abstain."""
    probability = np.asarray(probabilities, dtype=float)
    if isinstance(policy, dict):
        queue_policies = policy["queues"]
        positive_threshold = float(policy["positive_threshold"])
    else:
        queue_policies = policy.queues
        positive_threshold = policy.positive_threshold
    decisions: list[str] = []
    for row in probability:
        eligible: list[tuple[float, str]] = []
        for name, queue_policy in queue_policies.items():
            threshold = (
                float(queue_policy["threshold"])
                if isinstance(queue_policy, dict)
                else queue_policy.threshold
            )
            score = row[class_names.index(name)]
            if score >= threshold:
                eligible.append((score / threshold, f"{name}_candidate"))
        if eligible:
            decisions.append(max(eligible)[1])
        elif row[class_names.index("positive")] >= positive_threshold:
            decisions.append("positive")
        else:
            decisions.append("abstain")
    return np.asarray(decisions, dtype=object)


def evaluate_triage(
    y_true: np.ndarray,
    decisions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=int)
    decisions = np.asarray(decisions, dtype=object)
    metrics: dict[str, Any] = {"queues": {}}
    for name in ("negative", "neutral"):
        selected = decisions == f"{name}_candidate"
        truth = actual == class_names.index(name)
        true_positive = int(np.sum(selected & truth))
        metrics["queues"][name] = {
            "rows": int(selected.sum()),
            "queue_fraction": float(selected.mean()),
            "precision": float(true_positive / selected.sum()) if selected.any() else 0.0,
            "recall": float(true_positive / truth.sum()) if truth.any() else 0.0,
        }
    positive_selected = decisions == "positive"
    positive_truth = actual == class_names.index("positive")
    positive_true = int(np.sum(positive_selected & positive_truth))
    metrics["positive"] = {
        "rows": int(positive_selected.sum()),
        "coverage": float(positive_selected.mean()),
        "precision": (
            float(positive_true / positive_selected.sum()) if positive_selected.any() else 0.0
        ),
        "recall": float(positive_true / positive_truth.sum()),
    }
    metrics["abstain"] = {
        "rows": int(np.sum(decisions == "abstain")),
        "fraction": float(np.mean(decisions == "abstain")),
    }
    metrics["review_queue_rows"] = int(
        np.sum(np.isin(decisions, ["negative_candidate", "neutral_candidate"]))
    )
    return metrics
