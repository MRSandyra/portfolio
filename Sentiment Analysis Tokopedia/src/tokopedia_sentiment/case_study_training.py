"""Full three-class portfolio benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from tokopedia_sentiment.backtesting import rolling_temporal_product_backtest
from tokopedia_sentiment.case_study_data import (
    CLASS_NAMES,
    CaseStudySplits,
    make_case_study_split,
    prepare_case_study_reviews,
    split_audit,
)
from tokopedia_sentiment.case_study_evaluation import (
    bootstrap_confidence_intervals,
    evaluate_multiclass,
    evaluate_slices,
    write_case_study_report,
)
from tokopedia_sentiment.case_study_modeling import (
    DEFAULT_CANDIDATES,
    ModelSelection,
    compare_models,
    fit_calibrated_model,
)
from tokopedia_sentiment.constants import RANDOM_SEED
from tokopedia_sentiment.data import sha256_file
from tokopedia_sentiment.decision import (
    ThresholdSelection,
    predict_with_thresholds,
    select_minority_thresholds,
)
from tokopedia_sentiment.modeling import save_artifact, set_reproducible_seed
from tokopedia_sentiment.monitoring import compute_drift_report
from tokopedia_sentiment.triage import (
    TriagePolicy,
    apply_triage_policy,
    evaluate_triage,
    select_triage_policy,
)
from tokopedia_sentiment.two_stage import (
    ArchitectureComparison,
    compare_architectures,
    fit_architecture,
    fit_two_stage_model,
)

DATASET_SOURCE = "https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025"


@dataclass(frozen=True)
class CaseStudyTrainingConfig:
    data_path: Path
    artifact_path: Path
    report_directory: Path
    source_uri: str = DATASET_SOURCE
    split_strategy: str = "temporal"
    target_minority_recall: float = 0.70
    bootstrap_iterations: int = 500
    cv_folds: int = 3
    cv_repeats: int = 2
    seed: int = RANDOM_SEED
    candidates: tuple[tuple[str, float], ...] = DEFAULT_CANDIDATES
    run_group_holdout: bool = True
    compare_two_stage: bool = True
    run_rolling_backtest: bool = True


def _select_architecture(
    splits: CaseStudySplits, selection: ModelSelection, compare_two_stage: bool
) -> tuple[str, ArchitectureComparison | None, Any, np.ndarray]:
    single_model = fit_calibrated_model(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        model_name=selection.model_name,
        parameter=selection.parameter,
    )
    validation_text = splits.validation["text"].tolist()
    if not compare_two_stage:
        return (
            "single_stage",
            None,
            single_model,
            single_model.predict_proba(validation_text),
        )
    two_stage = fit_two_stage_model(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        parameter=selection.parameter,
    )
    comparison = compare_architectures(
        single_model,
        two_stage,
        validation_text,
        splits.validation["label"].tolist(),
    )
    validation_model = (
        two_stage if comparison.selected_architecture == "two_stage" else single_model
    )
    return (
        comparison.selected_architecture,
        comparison,
        validation_model,
        validation_model.predict_proba(validation_text),
    )


def _fit_final_model(splits: CaseStudySplits, selection: ModelSelection, architecture: str) -> Any:
    train_validation = pd.concat([splits.train, splits.validation], ignore_index=True)
    return fit_architecture(
        architecture,
        train_validation["text"].tolist(),
        train_validation["label"].tolist(),
        single_model_name=selection.model_name,
        parameter=selection.parameter,
    )


def _evaluate_split(
    splits: CaseStudySplits,
    selection: ModelSelection,
    architecture: str,
    target_recall: float,
) -> tuple[Any, ThresholdSelection, TriagePolicy, dict[str, Any], Any, Any, dict[str, Any]]:
    validation_model = fit_architecture(
        architecture,
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        single_model_name=selection.model_name,
        parameter=selection.parameter,
    )
    validation_probability = validation_model.predict_proba(splits.validation["text"].tolist())
    thresholds = select_minority_thresholds(
        splits.validation["label"].to_numpy(),
        validation_probability,
        CLASS_NAMES,
        target_recall=target_recall,
    )
    triage_policy = select_triage_policy(
        splits.validation["label"].to_numpy(), validation_probability, CLASS_NAMES
    )
    model = _fit_final_model(splits, selection, architecture)
    probability = model.predict_proba(splits.test["text"].tolist())
    predicted = predict_with_thresholds(
        probability,
        CLASS_NAMES,
        thresholds.thresholds,
        thresholds.majority_class,
    )
    metrics = evaluate_multiclass(
        splits.test["label"].to_numpy(), probability, predicted, CLASS_NAMES
    )
    triage_decisions = apply_triage_policy(probability, CLASS_NAMES, triage_policy)
    triage_metrics = evaluate_triage(splits.test["label"].to_numpy(), triage_decisions, CLASS_NAMES)
    return (
        model,
        thresholds,
        triage_policy,
        metrics,
        probability,
        predicted,
        triage_metrics,
    )


def _repeated_architecture_group_cv(
    frame: pd.DataFrame,
    selection: ModelSelection,
    architecture: str,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    x = frame["text"].to_numpy(dtype=object)
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["group"].to_numpy()
    results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(x, y, groups=groups), start=1
        ):
            model = fit_architecture(
                architecture,
                x[train_idx].tolist(),
                y[train_idx].tolist(),
                single_model_name=selection.model_name,
                parameter=selection.parameter,
            )
            predicted = model.predict(x[validation_idx].tolist())
            recalls = recall_score(
                y[validation_idx], predicted, labels=[0, 1, 2], average=None, zero_division=0
            )
            results.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "macro_f1": float(f1_score(y[validation_idx], predicted, average="macro")),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y[validation_idx], predicted)
                    ),
                    "per_class_recall": {
                        name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)
                    },
                }
            )
    return {
        "strategy": f"{repeats}x{folds} repeated stratified product-group CV",
        "architecture": architecture,
        "folds": results,
        "summary": {
            metric: {
                "mean": float(np.mean([result[metric] for result in results])),
                "std": float(np.std([result[metric] for result in results], ddof=1)),
            }
            for metric in ("macro_f1", "balanced_accuracy")
        },
        "per_class_recall": {
            name: {
                "mean": float(np.mean([result["per_class_recall"][name] for result in results])),
                "std": float(
                    np.std([result["per_class_recall"][name] for result in results], ddof=1)
                ),
            }
            for name in CLASS_NAMES
        },
    }


def train_case_study(config: CaseStudyTrainingConfig) -> dict[str, Any]:
    """Run selection, triage, temporal/group/backtest evaluation, CI, and slices."""
    set_reproducible_seed(config.seed)
    raw = pd.read_csv(config.data_path)
    prepared, preparation = prepare_case_study_reviews(raw)
    splits = make_case_study_split(prepared, strategy=config.split_strategy, seed=config.seed)

    selection = compare_models(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        splits.validation["text"].tolist(),
        splits.validation["label"].tolist(),
        candidates=config.candidates,
    )
    architecture, architecture_comparison, _, _ = _select_architecture(
        splits, selection, config.compare_two_stage
    )
    (
        model,
        thresholds,
        triage_policy,
        test_metrics,
        test_probability,
        test_predicted,
        triage_metrics,
    ) = _evaluate_split(splits, selection, architecture, config.target_minority_recall)

    def decision(probability: Any) -> Any:
        return predict_with_thresholds(
            probability, CLASS_NAMES, thresholds.thresholds, thresholds.majority_class
        )

    confidence_intervals = bootstrap_confidence_intervals(
        splits.test["label"].to_numpy(),
        test_probability,
        decision,
        CLASS_NAMES,
        iterations=config.bootstrap_iterations,
        seed=config.seed,
    )
    slices = evaluate_slices(splits.test, test_predicted, CLASS_NAMES)
    train_validation = pd.concat([splits.train, splits.validation], ignore_index=True)
    cross_validation = _repeated_architecture_group_cv(
        train_validation,
        selection,
        architecture,
        folds=config.cv_folds,
        repeats=config.cv_repeats,
        seed=config.seed,
    )

    group_holdout: dict[str, Any] | None = None
    if config.run_group_holdout:
        group_splits = make_case_study_split(prepared, strategy="group", seed=config.seed)
        (
            _,
            group_thresholds,
            group_triage_policy,
            group_metrics,
            _,
            _,
            group_triage_metrics,
        ) = _evaluate_split(group_splits, selection, architecture, config.target_minority_recall)
        group_holdout = {
            "split": split_audit(group_splits),
            "decision_policy": group_thresholds.to_dict(),
            "triage_policy": group_triage_policy.to_dict(),
            "metrics": group_metrics,
            "triage_metrics": group_triage_metrics,
        }

    rolling_backtest = (
        rolling_temporal_product_backtest(
            prepared,
            architecture=architecture,
            single_model_name=selection.model_name,
            parameter=selection.parameter,
            target_minority_recall=config.target_minority_recall,
        )
        if config.run_rolling_backtest
        else None
    )
    drift = compute_drift_report(train_validation, splits.test)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "source_uri": config.source_uri,
            "source_file_name": config.data_path.name,
            "source_sha256": sha256_file(config.data_path),
            "license_note": "Educational & Research Purpose; verify current Kaggle terms",
            "label_definition": {
                "negative": "rating 1-2 proxy label",
                "neutral": "rating 3 proxy label",
                "positive": "rating 4-5 proxy label",
            },
            "preparation": preparation.to_dict(),
        },
        "seed": config.seed,
        "primary_split": split_audit(splits),
        "model_selection": selection.to_dict(),
        "architecture_comparison": (
            architecture_comparison.to_dict() if architecture_comparison else None
        ),
        "selected_architecture": architecture,
        "decision_policy": thresholds.to_dict(),
        "triage_policy": triage_policy.to_dict(),
        "test_metrics": test_metrics,
        "triage_metrics": triage_metrics,
        "bootstrap_confidence_intervals": confidence_intervals,
        "slice_metrics": slices,
        "drift": drift,
        "repeated_product_group_cv": cross_validation,
        "product_group_holdout": group_holdout,
        "rolling_temporal_product_backtest": rolling_backtest,
    }
    metadata = {
        "created_at_utc": report["created_at_utc"],
        "seed": config.seed,
        "dataset": report["dataset"],
        "class_names": CLASS_NAMES,
        "model": {
            "name": selection.model_name,
            "parameter": selection.parameter,
            "architecture": architecture,
            "probabilities_calibrated": True,
        },
        "decision_policy": thresholds.to_dict(),
        "triage_policy": triage_policy.to_dict(),
        "test_metrics": test_metrics,
        "triage_metrics": triage_metrics,
    }
    save_artifact(model, metadata, config.artifact_path)
    write_case_study_report(
        report,
        config.report_directory,
        y_true=splits.test["label"].to_numpy(),
        probabilities=test_probability,
        class_names=CLASS_NAMES,
    )
    return report
