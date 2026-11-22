"""Multiclass metrics, confidence intervals, calibration, and slice diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from tokopedia_sentiment.evaluation import expected_calibration_error


def evaluate_multiclass(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(predicted, dtype=int)
    labels = list(range(len(class_names)))
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    one_hot = np.eye(len(class_names))[actual]
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted")),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "per_class": {
            name: {
                metric: float(report[name][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for name in class_names
        },
        "probability_quality": {
            "log_loss": float(log_loss(actual, probability, labels=labels)),
            "multiclass_brier_score": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "per_class": {},
        },
        "average_precision": {},
    }
    for index, name in enumerate(class_names):
        binary_actual = (actual == index).astype(int)
        result["average_precision"][name] = float(
            average_precision_score(binary_actual, probability[:, index])
        )
        result["probability_quality"]["per_class"][name] = {
            "brier_score": float(brier_score_loss(binary_actual, probability[:, index])),
            "expected_calibration_error_10_bins": expected_calibration_error(
                binary_actual, probability[:, index]
            ),
        }
    result["macro_average_precision"] = float(np.mean(list(result["average_precision"].values())))
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(actual, probability, labels=labels, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["roc_auc_ovr_macro"] = None
    return result


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    decision_function: Callable[[np.ndarray], np.ndarray],
    class_names: list[str],
    *,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified bootstrap 95% intervals for aggregate and class recall metrics."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(actual == index) for index in range(len(class_names))]
    values: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        **{f"recall_{name}": [] for name in class_names},
    }
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        sampled_actual = actual[sampled]
        sampled_predicted = decision_function(probability[sampled])
        recalls = recall_score(
            sampled_actual,
            sampled_predicted,
            labels=list(range(len(class_names))),
            average=None,
            zero_division=0,
        )
        values["accuracy"].append(float(accuracy_score(sampled_actual, sampled_predicted)))
        values["balanced_accuracy"].append(
            float(balanced_accuracy_score(sampled_actual, sampled_predicted))
        )
        values["macro_f1"].append(
            float(f1_score(sampled_actual, sampled_predicted, average="macro"))
        )
        for index, name in enumerate(class_names):
            values[f"recall_{name}"].append(float(recalls[index]))
    result: dict[str, Any] = {
        "method": f"stratified bootstrap, {iterations} iterations, 95% percentile interval"
    }
    result.update(
        {
            metric: {
                "lower": float(np.percentile(samples, 2.5)),
                "median": float(np.percentile(samples, 50)),
                "upper": float(np.percentile(samples, 97.5)),
            }
            for metric, samples in values.items()
        }
    )
    return result


def _slice_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    class_names: list[str],
) -> dict[str, Any] | None:
    if int(mask.sum()) < 10:
        return None
    slice_actual = actual[mask]
    slice_predicted = predicted[mask]
    recalls = recall_score(
        slice_actual,
        slice_predicted,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    return {
        "rows": int(mask.sum()),
        "accuracy": float(accuracy_score(slice_actual, slice_predicted)),
        "macro_f1": float(
            f1_score(
                slice_actual,
                slice_predicted,
                labels=list(range(len(class_names))),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(class_names)},
    }


def evaluate_slices(
    frame: pd.DataFrame, predicted: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Evaluate ratings, categories, and common Indonesian text edge cases."""
    actual = frame["label"].to_numpy(dtype=int)
    raw = frame["raw_text"].astype(str)
    normalized = frame["text"].astype(str)
    token_count = normalized.str.split().str.len()
    masks: dict[str, np.ndarray] = {}
    for rating in sorted(frame["rating"].unique()):
        masks[f"rating_{rating}"] = frame["rating"].eq(rating).to_numpy()
    for category in sorted(frame["category"].unique()):
        masks[f"category:{category}"] = frame["category"].eq(category).to_numpy()
    masks.update(
        {
            "short_text_1_to_3_tokens": token_count.between(1, 3).to_numpy(),
            "long_text_30_plus_tokens": token_count.ge(30).to_numpy(),
            "contains_negation": normalized.str.contains(
                r"\b(?:tidak|belum|bukan|jangan)\b", regex=True
            ).to_numpy(),
            "contains_slang": raw.str.casefold()
            .str.contains(r"\b(?:gak|gk|nggak|bgt|udh|yg)\b", regex=True)
            .to_numpy(),
            "mixed_positive_negative_cues": (
                normalized.str.contains(r"\b(?:bagus|enak|mantap|puas|suka)\b", regex=True)
                & normalized.str.contains(
                    r"\b(?:buruk|kecewa|rusak|tidak|basi|jelek)\b", regex=True
                )
            ).to_numpy(),
        }
    )
    results: dict[str, Any] = {}
    for name, mask in masks.items():
        metrics = _slice_metrics(actual, predicted, mask, class_names)
        if metrics is not None:
            results[name] = metrics
    return results


def write_case_study_report(
    report: dict[str, Any],
    output_directory: str | Path,
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    matrix = np.asarray(report["test_metrics"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (untouched test)",
    )
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    actual = np.asarray(y_true, dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for index, name in enumerate(class_names):
        observed, predicted = calibration_curve(
            (actual == index).astype(int), probabilities[:, index], n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed class frequency",
        title="One-vs-rest reliability diagram",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "calibration_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    for index, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            (actual == index).astype(int), probabilities[:, index]
        )
        axis.plot(recall, precision, label=name)
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="One-vs-rest precision-recall curves",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "precision_recall_curves.png", dpi=150)
    plt.close(figure)
