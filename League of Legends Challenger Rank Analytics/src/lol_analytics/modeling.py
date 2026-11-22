"""Train/evaluate an intentionally small, inspectable model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .metrics import classification_metrics


@dataclass
class ModelResult:
    model: Pipeline
    metrics: dict[str, float]
    test_predictions: pd.DataFrame


def make_model(*, regularization_c: float = 0.1) -> Pipeline:
    """All learned preprocessing is fitted inside the training-only pipeline."""
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    C=regularization_c,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "win",
    *,
    regularization_c: float = 0.1,
) -> ModelResult:
    model = fit_model(
        train,
        features,
        target=target,
        regularization_c=regularization_c,
    )
    return evaluate_model(model, test, features, target=target)


def fit_model(
    train: pd.DataFrame,
    features: list[str],
    target: str = "win",
    *,
    regularization_c: float = 0.1,
) -> Pipeline:
    """Fit preprocessing and classifier on training rows only."""
    if train[target].nunique() < 2:
        raise ValueError("Training data must contain both target classes")
    model = make_model(regularization_c=regularization_c)
    model.fit(train[features], train[target])
    return model


def evaluate_model(
    model: Pipeline,
    test: pd.DataFrame,
    features: list[str],
    target: str = "win",
) -> ModelResult:
    """Evaluate an already-fitted model without mutating or refitting it."""
    if test[target].nunique() < 2:
        raise ValueError("Test data must contain both target classes to compute ROC-AUC")
    probability = model.predict_proba(test[features])[:, 1]
    metrics = classification_metrics(test[target].to_numpy(), probability)
    provenance = [
        column
        for column in ["matchId", "game_start_timestamp_ms", "match_order", target]
        if column in test
    ]
    predictions = test[provenance].copy()
    predictions["win_probability"] = probability
    return ModelResult(model=model, metrics=metrics, test_predictions=predictions)

