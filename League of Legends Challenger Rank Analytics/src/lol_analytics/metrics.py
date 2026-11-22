"""Binary classification metrics with explicit input validation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_probability, dtype=float)
    if truth.shape != probability.shape:
        raise ValueError("y_true and y_probability must have the same shape")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("ROC-AUC requires both target classes")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
    }

