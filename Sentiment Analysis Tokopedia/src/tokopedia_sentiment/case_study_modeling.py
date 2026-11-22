"""Imbalance-aware model comparison and repeated product-group validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from tokopedia_sentiment.case_study_data import CLASS_NAMES
from tokopedia_sentiment.preprocessing import normalize_indonesian_text


@dataclass(frozen=True)
class ModelSelection:
    model_name: str
    parameter: float
    validation_macro_f1: float
    validation_balanced_accuracy: float
    validation_minority_recall: float
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CANDIDATES: tuple[tuple[str, float], ...] = (
    ("count_multinomial_nb", 0.1),
    ("count_multinomial_nb", 1.0),
    ("tfidf_complement_nb", 0.1),
    ("tfidf_complement_nb", 1.0),
    ("tfidf_logreg_balanced", 0.5),
    ("tfidf_logreg_balanced", 1.0),
)


def _word_count() -> CountVectorizer:
    return CountVectorizer(
        preprocessor=normalize_indonesian_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
    )


def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_indonesian_text,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=80_000,
    )


def build_case_study_model(model_name: str, parameter: float) -> Pipeline:
    if model_name == "count_multinomial_nb":
        return Pipeline(
            [("features", _word_count()), ("classifier", MultinomialNB(alpha=parameter))]
        )
    if model_name == "tfidf_complement_nb":
        return Pipeline(
            [("features", _word_tfidf()), ("classifier", ComplementNB(alpha=parameter))]
        )
    if model_name == "tfidf_logreg_balanced":
        features = FeatureUnion(
            [
                ("word", _word_tfidf()),
                (
                    "character",
                    TfidfVectorizer(
                        preprocessor=normalize_indonesian_text,
                        lowercase=False,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        classifier = LogisticRegression(
            C=parameter,
            class_weight="balanced",
            max_iter=1_000,
            solver="lbfgs",
            random_state=42,
        )
        return Pipeline([("features", features), ("classifier", classifier)])
    raise ValueError(f"unknown model: {model_name}")


def compare_models(
    train_text: list[str],
    train_labels: list[int],
    validation_text: list[str],
    validation_labels: list[int],
    *,
    candidates: tuple[tuple[str, float], ...] = DEFAULT_CANDIDATES,
) -> ModelSelection:
    actual = np.asarray(validation_labels, dtype=int)
    majority_id = int(np.argmax(np.bincount(np.asarray(train_labels), minlength=3)))
    minority_ids = [index for index in range(3) if index != majority_id]
    results: list[dict[str, Any]] = []
    for model_name, parameter in candidates:
        model = build_case_study_model(model_name, parameter)
        model.fit(train_text, train_labels)
        predicted = model.predict(validation_text)
        recalls = recall_score(actual, predicted, labels=[0, 1, 2], average=None, zero_division=0)
        results.append(
            {
                "model_name": model_name,
                "parameter": parameter,
                "macro_f1": float(f1_score(actual, predicted, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "minority_recall": float(np.mean(recalls[minority_ids])),
                "per_class_recall": {
                    CLASS_NAMES[index]: float(recalls[index]) for index in range(3)
                },
            }
        )
    best = max(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["minority_recall"],
            item["balanced_accuracy"],
        ),
    )
    return ModelSelection(
        model_name=best["model_name"],
        parameter=float(best["parameter"]),
        validation_macro_f1=float(best["macro_f1"]),
        validation_balanced_accuracy=float(best["balanced_accuracy"]),
        validation_minority_recall=float(best["minority_recall"]),
        candidates=results,
    )


def fit_calibrated_model(
    text: list[str], labels: list[int], *, model_name: str, parameter: float
) -> Any:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3)
    folds = min(3, int(counts.min()))
    if folds < 2:
        raise ValueError("calibration requires at least two rows per class")
    base = build_case_study_model(model_name, parameter)
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=folds).fit(text, labels)


def repeated_group_cross_validation(
    text: list[str],
    labels: list[int],
    groups: list[object],
    *,
    model_name: str,
    parameter: float,
    folds: int = 3,
    repeats: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure stability while preventing the same product from crossing folds."""
    x = np.asarray(text, dtype=object)
    y = np.asarray(labels, dtype=int)
    group_array = np.asarray(groups)
    fold_results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(x, y, groups=group_array), start=1
        ):
            model = build_case_study_model(model_name, parameter)
            model.fit(x[train_idx].tolist(), y[train_idx].tolist())
            predicted = model.predict(x[validation_idx].tolist())
            recalls = recall_score(
                y[validation_idx], predicted, labels=[0, 1, 2], average=None, zero_division=0
            )
            fold_results.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "macro_f1": float(f1_score(y[validation_idx], predicted, average="macro")),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y[validation_idx], predicted)
                    ),
                    "per_class_recall": {
                        CLASS_NAMES[index]: float(recalls[index]) for index in range(3)
                    },
                }
            )
    return {
        "strategy": f"{repeats}x{folds} repeated stratified product-group CV",
        "folds": fold_results,
        "summary": {
            metric: {
                "mean": float(np.mean([result[metric] for result in fold_results])),
                "std": float(np.std([result[metric] for result in fold_results], ddof=1)),
            }
            for metric in ("macro_f1", "balanced_accuracy")
        },
        "per_class_recall": {
            name: {
                "mean": float(
                    np.mean([result["per_class_recall"][name] for result in fold_results])
                ),
                "std": float(
                    np.std([result["per_class_recall"][name] for result in fold_results], ddof=1)
                ),
            }
            for name in CLASS_NAMES
        },
    }
