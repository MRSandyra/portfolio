"""Model construction, validation-set selection, and artifact persistence."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from tokopedia_sentiment.constants import RANDOM_SEED
from tokopedia_sentiment.preprocessing import normalize_indonesian_text

ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectionResult:
    alpha: float
    validation_macro_f1: float
    candidates: list[dict[str, float]]


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python and NumPy for repeatable local runs."""
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline(*, alpha: float = 1.0) -> Pipeline:
    """Build a count n-gram + MultinomialNB baseline.

    Normalization lives inside the estimator so inference cannot accidentally skip it.
    Bigrams preserve phrases such as ``tidak enak`` that unigrams lose.
    """
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    preprocessor=normalize_indonesian_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    max_features=50_000,
                ),
            ),
            ("classifier", MultinomialNB(alpha=alpha)),
        ]
    )


def select_alpha(
    train_text: list[str],
    train_labels: list[int],
    validation_text: list[str],
    validation_labels: list[int],
    *,
    candidates: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
) -> SelectionResult:
    """Select smoothing by validation macro-F1; never inspect the test set."""
    scores: list[dict[str, float]] = []
    for alpha in candidates:
        model = build_pipeline(alpha=alpha)
        model.fit(train_text, train_labels)
        predictions = model.predict(validation_text)
        score = float(f1_score(validation_labels, predictions, average="macro"))
        scores.append({"alpha": float(alpha), "macro_f1": score})

    # Deterministic tie break: prefer the larger smoothing value.
    best = max(scores, key=lambda item: (item["macro_f1"], item["alpha"]))
    return SelectionResult(
        alpha=best["alpha"],
        validation_macro_f1=best["macro_f1"],
        candidates=scores,
    )


def fit_final_model(
    text: list[str],
    labels: list[int],
    *,
    alpha: float,
    calibrate: bool = False,
) -> Any:
    """Fit on train+validation after selection, optionally calibrating probabilities."""
    base_model = build_pipeline(alpha=alpha)
    if not calibrate:
        return base_model.fit(text, labels)

    class_counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    calibration_folds = min(5, int(class_counts.min()))
    if calibration_folds < 2:
        raise ValueError("calibration needs at least two examples in each class")
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_folds,
    )
    return calibrated.fit(text, labels)


def save_artifact(model: Any, metadata: dict[str, Any], path: str | Path) -> Path:
    """Save model plus auditable metadata as a versioned joblib bundle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(bundle, destination)
    return destination


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a trusted local artifact and validate its minimal schema.

    Joblib/pickle artifacts can execute code while loading. Do not load untrusted files.
    """
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported or invalid model artifact")
    if "model" not in bundle or "metadata" not in bundle:
        raise ValueError("model artifact is missing required fields")
    return bundle
