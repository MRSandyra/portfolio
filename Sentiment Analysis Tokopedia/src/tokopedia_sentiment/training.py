"""End-to-end training orchestration with an untouched test split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tokopedia_sentiment.constants import DEFAULT_DATASET_URL, RANDOM_SEED
from tokopedia_sentiment.data import (
    load_reviews,
    prepare_reviews,
    sha256_file,
    stratified_train_validation_test_split,
)
from tokopedia_sentiment.evaluation import evaluate_probabilities, write_evaluation_report
from tokopedia_sentiment.modeling import (
    fit_final_model,
    save_artifact,
    select_alpha,
    set_reproducible_seed,
)


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    artifact_path: Path
    report_directory: Path
    source_uri: str = DEFAULT_DATASET_URL
    text_column: str = "Review"
    rating_column: str = "Rating"
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = RANDOM_SEED
    calibrate: bool = False


def train(config: TrainingConfig) -> dict[str, Any]:
    """Train, select on validation, and report exactly once on held-out test data."""
    set_reproducible_seed(config.seed)
    raw = load_reviews(config.data_path)
    prepared, preparation_summary = prepare_reviews(
        raw,
        text_column=config.text_column,
        rating_column=config.rating_column,
    )
    splits = stratified_train_validation_test_split(
        prepared,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.seed,
    )

    selection = select_alpha(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        splits.validation["text"].tolist(),
        splits.validation["label"].tolist(),
    )
    train_validation = pd.concat([splits.train, splits.validation], ignore_index=True)
    model = fit_final_model(
        train_validation["text"].tolist(),
        train_validation["label"].tolist(),
        alpha=selection.alpha,
        calibrate=config.calibrate,
    )

    positive_probability = model.predict_proba(splits.test["text"].tolist())[:, 1]
    metrics = evaluate_probabilities(splits.test["label"].to_numpy(), positive_probability)
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "label_definition": {"0": "negative: rating 1-3", "1": "positive: rating 4-5"},
        "source_uri": config.source_uri,
        "source_file_name": config.data_path.name,
        "source_sha256": sha256_file(config.data_path),
        "preparation": preparation_summary.to_dict(),
        "split": {
            "strategy": "stratified train/validation/test after normalization and deduplication",
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "rows": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
        },
        "model": {
            "type": "CountVectorizer(1,2-grams) + MultinomialNB",
            "selected_alpha": selection.alpha,
            "selection_metric": "validation macro-F1",
            "validation_macro_f1": selection.validation_macro_f1,
            "candidate_scores": selection.candidates,
            "probabilities_calibrated": config.calibrate,
        },
        "test_metrics": metrics,
    }
    save_artifact(model, metadata, config.artifact_path)
    write_evaluation_report(
        metrics,
        config.report_directory,
        y_true=splits.test["label"].to_numpy(),
        positive_probability=positive_probability,
    )
    return metadata


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """JSON-friendly representation useful to callers and tests."""
    result = asdict(config)
    for key in ("data_path", "artifact_path", "report_directory"):
        result[key] = str(result[key])
    return result
