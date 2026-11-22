"""Validation-only decision-threshold selection for rare sentiment classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class ThresholdSelection:
    thresholds: dict[str, float]
    majority_class: str
    target_recall: float
    target_met: bool
    validation_macro_f1: float
    validation_balanced_accuracy: float
    validation_recall: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def predict_with_thresholds(
    probabilities: np.ndarray,
    class_names: list[str],
    thresholds: dict[str, float],
    majority_class: str,
) -> np.ndarray:
    """Apply one-vs-rest thresholds, resolving ties by normalized probability."""
    probabilities = np.asarray(probabilities, dtype=float)
    majority_id = class_names.index(majority_class)
    predictions = np.full(len(probabilities), majority_id, dtype=int)
    minority_ids = [index for index, name in enumerate(class_names) if name != majority_class]
    for row_index, row in enumerate(probabilities):
        eligible: list[tuple[float, int]] = []
        for class_id in minority_ids:
            threshold = thresholds[class_names[class_id]]
            if row[class_id] >= threshold:
                eligible.append((row[class_id] / threshold, class_id))
        if eligible:
            predictions[row_index] = max(eligible)[1]
    return predictions


def select_minority_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    *,
    target_recall: float = 0.70,
) -> ThresholdSelection:
    """Choose thresholds on validation data only.

    Among policies meeting the recall target for every minority class, macro-F1 is
    maximized. If the target is infeasible, the best minimum minority recall wins.
    """
    actual = np.asarray(y_true, dtype=int)
    counts = np.bincount(actual, minlength=len(class_names))
    majority_id = int(np.argmax(counts))
    majority_class = class_names[majority_id]
    minority_names = [name for name in class_names if name != majority_class]
    grid = np.unique(np.concatenate(([0.01, 0.02, 0.03], np.linspace(0.05, 0.50, 19))))

    best_key: tuple[float, ...] | None = None
    best_payload: tuple[dict[str, float], np.ndarray, dict[str, float], bool] | None = None
    labels = list(range(len(class_names)))
    for values in product(grid, repeat=len(minority_names)):
        thresholds = dict(zip(minority_names, (float(value) for value in values), strict=True))
        predicted = predict_with_thresholds(probabilities, class_names, thresholds, majority_class)
        recall_values = recall_score(
            actual, predicted, labels=labels, average=None, zero_division=0
        )
        recalls = {
            class_names[index]: float(recall_values[index]) for index in range(len(class_names))
        }
        minority_recalls = [recalls[name] for name in minority_names]
        target_met = all(value >= target_recall for value in minority_recalls)
        macro_f1 = float(f1_score(actual, predicted, labels=labels, average="macro"))
        balanced = float(balanced_accuracy_score(actual, predicted))
        # Feasible policies always outrank infeasible ones. Within each group, preserve
        # useful precision through macro-F1 before using recall/balanced accuracy ties.
        key = (
            float(target_met),
            macro_f1 if target_met else min(minority_recalls),
            balanced,
            macro_f1,
            sum(values),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_payload = (thresholds, predicted, recalls, target_met)

    if best_payload is None:
        raise RuntimeError("threshold search produced no candidates")
    thresholds, predicted, recalls, target_met = best_payload
    return ThresholdSelection(
        thresholds=thresholds,
        majority_class=majority_class,
        target_recall=target_recall,
        target_met=target_met,
        validation_macro_f1=float(f1_score(actual, predicted, average="macro")),
        validation_balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
        validation_recall=recalls,
    )
