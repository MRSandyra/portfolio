"""Classification and probability-quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: list[int] | np.ndarray,
    positive_probability: list[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error for the positive class."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(positive_probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(actual[mask].mean()) - float(probability[mask].mean())
            )
    return error


def evaluate_probabilities(
    y_true: list[int] | np.ndarray,
    positive_probability: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Return per-class, aggregate, confusion, and calibration diagnostics."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(positive_probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        "per_class": {
            name: {
                metric: float(report[name][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for name in ("negative", "positive")
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "probability_quality": {
            "brier_score": float(brier_score_loss(actual, probability)),
            "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
            "expected_calibration_error_10_bins": expected_calibration_error(actual, probability),
        },
    }
    if np.unique(actual).size == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
    return result


def write_evaluation_report(
    report: dict[str, Any],
    output_directory: str | Path,
    *,
    y_true: list[int] | np.ndarray,
    positive_probability: list[float] | np.ndarray,
) -> None:
    """Write JSON plus confusion-matrix and reliability plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    matrix = np.asarray(report["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (held-out test)",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    fraction_positive, mean_predicted = calibration_curve(
        y_true, positive_probability, n_bins=10, strategy="uniform"
    )
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axis.plot(mean_predicted, fraction_positive, marker="o", label="model")
    axis.set(
        xlabel="Mean predicted P(positive)",
        ylabel="Observed positive fraction",
        title="Reliability diagram (held-out test)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "calibration_curve.png", dpi=150)
    plt.close(figure)
