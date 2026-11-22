"""Rolling chronological backtests with seen/unseen-product slices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from tokopedia_sentiment.case_study_data import CLASS_NAMES
from tokopedia_sentiment.decision import predict_with_thresholds, select_minority_thresholds
from tokopedia_sentiment.monitoring import compute_drift_report
from tokopedia_sentiment.triage import apply_triage_policy, evaluate_triage, select_triage_policy
from tokopedia_sentiment.two_stage import fit_architecture


def _classification_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    recalls = recall_score(actual, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "rows": len(actual),
        "macro_f1": float(
            f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)},
        "support": {name: int(np.sum(actual == index)) for index, name in enumerate(CLASS_NAMES)},
    }


def rolling_temporal_product_backtest(
    frame: pd.DataFrame,
    *,
    architecture: str,
    single_model_name: str,
    parameter: float,
    test_starts: tuple[str, ...] = ("2024-07-01", "2025-01-01", "2025-07-01"),
    window_months: int = 6,
    target_minority_recall: float = 0.70,
) -> dict[str, Any]:
    """Tune on the prior window and test the next window, including unseen products."""
    windows: list[dict[str, Any]] = []
    for test_start_text in test_starts:
        test_start = pd.Timestamp(test_start_text)
        validation_start = test_start - pd.DateOffset(months=window_months)
        test_end = test_start + pd.DateOffset(months=window_months)
        train = frame.loc[frame["date"] < validation_start]
        validation = frame.loc[(frame["date"] >= validation_start) & (frame["date"] < test_start)]
        test = frame.loc[(frame["date"] >= test_start) & (frame["date"] < test_end)]
        if any(part["label"].nunique() < 3 for part in (train, validation, test)):
            continue

        validation_model = fit_architecture(
            architecture,
            train["text"].tolist(),
            train["label"].tolist(),
            single_model_name=single_model_name,
            parameter=parameter,
        )
        validation_probability = validation_model.predict_proba(validation["text"].tolist())
        thresholds = select_minority_thresholds(
            validation["label"].to_numpy(),
            validation_probability,
            CLASS_NAMES,
            target_recall=target_minority_recall,
        )
        triage_policy = select_triage_policy(
            validation["label"].to_numpy(), validation_probability, CLASS_NAMES
        )

        train_validation = pd.concat([train, validation], ignore_index=True)
        final_model = fit_architecture(
            architecture,
            train_validation["text"].tolist(),
            train_validation["label"].tolist(),
            single_model_name=single_model_name,
            parameter=parameter,
        )
        test_probability = final_model.predict_proba(test["text"].tolist())
        predicted = predict_with_thresholds(
            test_probability,
            CLASS_NAMES,
            thresholds.thresholds,
            thresholds.majority_class,
        )
        triage_decisions = apply_triage_policy(test_probability, CLASS_NAMES, triage_policy)
        known_products = set(train_validation["group"])
        unseen_mask = ~test["group"].isin(known_products).to_numpy()
        seen_mask = ~unseen_mask
        slices = {"all": _classification_summary(test["label"].to_numpy(), predicted)}
        for name, mask in (("seen_product", seen_mask), ("unseen_product", unseen_mask)):
            if int(mask.sum()) >= 10:
                slices[name] = _classification_summary(
                    test["label"].to_numpy()[mask], predicted[mask]
                )
        windows.append(
            {
                "train_end": (validation_start - pd.Timedelta(days=1)).date().isoformat(),
                "validation_range": [
                    validation_start.date().isoformat(),
                    (test_start - pd.Timedelta(days=1)).date().isoformat(),
                ],
                "test_range": [
                    test_start.date().isoformat(),
                    (test_end - pd.Timedelta(days=1)).date().isoformat(),
                ],
                "decision_policy": thresholds.to_dict(),
                "triage_policy": triage_policy.to_dict(),
                "classification": slices,
                "triage": evaluate_triage(test["label"].to_numpy(), triage_decisions, CLASS_NAMES),
                "drift": compute_drift_report(train_validation, test),
            }
        )
    macro_scores = [window["classification"]["all"]["macro_f1"] for window in windows]
    return {
        "method": "rolling six-month validation/test with seen and unseen product slices",
        "architecture": architecture,
        "windows": windows,
        "summary": {
            "macro_f1_mean": float(np.mean(macro_scores)) if macro_scores else None,
            "macro_f1_std": (
                float(np.std(macro_scores, ddof=1)) if len(macro_scores) > 1 else None
            ),
        },
    }
