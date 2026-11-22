"""Temporal validation and bootstrap uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import RANDOM_STATE
from .modeling import train_and_evaluate
from .split import DEFAULT_TIME_COLUMN


@dataclass(frozen=True)
class TemporalFold:
    train: pd.DataFrame
    test: pd.DataFrame
    fold: int


def expanding_window_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    order_column: str = DEFAULT_TIME_COLUMN,
    boundary_offset: int = 0,
) -> list[TemporalFold]:
    """Create expanding train windows followed by non-overlapping time blocks."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if order_column not in frame or frame[order_column].isna().any():
        raise ValueError(f"A complete {order_column} column is required")
    ordered = frame.sort_values([order_column, "matchId"], kind="stable").reset_index(drop=True)
    unique_times = np.sort(ordered[order_column].unique())
    if len(unique_times) < n_splits + 1:
        raise ValueError("Not enough unique timestamps for expanding-window validation")
    cut_indices = [
        int(round(len(unique_times) * index / (n_splits + 1))) + boundary_offset
        for index in range(1, n_splits + 1)
    ]
    if cut_indices[0] < 1 or cut_indices[-1] >= len(unique_times):
        raise ValueError("boundary_offset leaves an empty temporal partition")
    if len(set(cut_indices)) != len(cut_indices):
        raise ValueError("Not enough unique timestamps for shifted temporal boundaries")
    boundaries = [0, *cut_indices, len(unique_times)]
    folds: list[TemporalFold] = []
    for fold_index in range(1, n_splits + 1):
        test_times = unique_times[boundaries[fold_index] : boundaries[fold_index + 1]]
        test_start = test_times[0]
        test_end = test_times[-1]
        train = ordered.loc[ordered[order_column].lt(test_start)].copy()
        test = ordered.loc[ordered[order_column].between(test_start, test_end)].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal fold contains an empty partition")
        if train[order_column].max() >= test[order_column].min():
            raise AssertionError("Internal error: temporal fold overlaps")
        folds.append(TemporalFold(train=train, test=test, fold=fold_index))
    return folds


def repeated_boundary_offsets(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    repeats: int,
    max_shift_fraction: float = 0.4,
    order_column: str = DEFAULT_TIME_COLUMN,
) -> list[int]:
    """Return symmetric temporal-boundary shifts, always including the base split."""
    if repeats < 1 or repeats % 2 == 0:
        raise ValueError("repeats must be a positive odd integer")
    if not 0 < max_shift_fraction < 0.5:
        raise ValueError("max_shift_fraction must be between 0 and 0.5")
    if order_column not in frame or frame[order_column].isna().any():
        raise ValueError(f"A complete {order_column} column is required")
    if repeats == 1:
        return [0]
    unique_times = frame[order_column].nunique()
    nominal_block = unique_times // (n_splits + 1)
    max_shift = int(nominal_block * max_shift_fraction)
    if max_shift < (repeats - 1) // 2:
        raise ValueError("Not enough timestamps for the requested repeated boundary shifts")
    offsets = np.linspace(-max_shift, max_shift, repeats).round().astype(int).tolist()
    if len(set(offsets)) != repeats or 0 not in offsets:
        raise ValueError("Repeated boundary shifts must be unique and include zero")
    return offsets


def block_bootstrap_refit_auc_interval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    regularization_c: float,
    iterations: int = 200,
    block_size: int = 50,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    """Refit on resampled contiguous temporal blocks and score one fixed future block."""
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if block_size < 2 or block_size > len(train):
        raise ValueError("block_size must be between 2 and the training cohort size")
    ordered = train.sort_values([DEFAULT_TIME_COLUMN, "matchId"], kind="stable").reset_index(
        drop=True
    )
    blocks = [
        np.arange(start, min(start + block_size, len(ordered)))
        for start in range(0, len(ordered), block_size)
    ]
    rng = np.random.default_rng(random_state)
    samples = np.empty(iterations, dtype=float)
    redraws = 0
    for iteration in range(iterations):
        for _ in range(1_000):
            selected_blocks: list[np.ndarray] = []
            selected_rows = 0
            while selected_rows < len(ordered):
                block = blocks[int(rng.integers(0, len(blocks)))]
                selected_blocks.append(block)
                selected_rows += len(block)
            selected = np.concatenate(selected_blocks)[: len(ordered)]
            bootstrap_train = ordered.iloc[selected].copy()
            if bootstrap_train["win"].nunique() == 2:
                break
            redraws += 1
            redraws += 1
        else:
            raise RuntimeError("Could not draw a two-class temporal block bootstrap sample")
        result = train_and_evaluate(
            bootstrap_train,
            test,
            features,
            regularization_c=regularization_c,
        )
        samples[iteration] = result.metrics["roc_auc"]
    point = train_and_evaluate(
        ordered,
        test,
        features,
        regularization_c=regularization_c,
    ).metrics["roc_auc"]
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(point),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
        "block_size_matches": block_size,
        "temporal_blocks": len(blocks),
        "training_matches": len(ordered),
        "test_matches": len(test),
        "bootstrap_refits": iterations,
        "two_class_redraws": redraws,
    }


def _stratified_bootstrap_indices(
    truth: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    negative = np.flatnonzero(truth == 0)
    positive = np.flatnonzero(truth == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Bootstrap ROC-AUC requires both classes")
    sampled = np.concatenate(
        [
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ]
    )
    rng.shuffle(sampled)
    return sampled


def bootstrap_auc_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probability, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError("y_true and probability must have the same shape")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = _stratified_bootstrap_indices(truth, rng)
        samples[index] = roc_auc_score(truth[selected], scores[selected])
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(roc_auc_score(truth, scores)),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
    }


def paired_auc_delta_interval(
    y_true: np.ndarray,
    early_probability: np.ndarray,
    mid_probability: np.ndarray,
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    early = np.asarray(early_probability, dtype=float)
    mid = np.asarray(mid_probability, dtype=float)
    if truth.shape != early.shape or truth.shape != mid.shape:
        raise ValueError("Paired probabilities and y_true must have the same shape")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    rng = np.random.default_rng(random_state)
    deltas = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = _stratified_bootstrap_indices(truth, rng)
        deltas[index] = roc_auc_score(truth[selected], mid[selected]) - roc_auc_score(
            truth[selected], early[selected]
        )
    alpha = (1 - confidence) / 2
    estimate = roc_auc_score(truth, mid) - roc_auc_score(truth, early)
    return {
        "estimate_mid_minus_early": float(estimate),
        "lower": float(np.quantile(deltas, alpha)),
        "upper": float(np.quantile(deltas, 1 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
        "bootstrap_probability_mid_gt_early": float(np.mean(deltas > 0)),
    }


def independent_auc_delta_interval(
    first_truth: np.ndarray,
    first_probability: np.ndarray,
    second_truth: np.ndarray,
    second_probability: np.ndarray,
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    """Bootstrap an AUC difference for two independent evaluation cohorts."""
    first_y = np.asarray(first_truth, dtype=int)
    first_scores = np.asarray(first_probability, dtype=float)
    second_y = np.asarray(second_truth, dtype=int)
    second_scores = np.asarray(second_probability, dtype=float)
    if first_y.shape != first_scores.shape or second_y.shape != second_scores.shape:
        raise ValueError("Each truth and probability pair must have the same shape")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    deltas = np.empty(iterations, dtype=float)
    for index in range(iterations):
        first_selected = _stratified_bootstrap_indices(first_y, rng)
        second_selected = _stratified_bootstrap_indices(second_y, rng)
        deltas[index] = roc_auc_score(
            second_y[second_selected], second_scores[second_selected]
        ) - roc_auc_score(first_y[first_selected], first_scores[first_selected])
    alpha = (1 - confidence) / 2
    estimate = roc_auc_score(second_y, second_scores) - roc_auc_score(
        first_y, first_scores
    )
    return {
        "estimate_second_minus_first": float(estimate),
        "lower": float(np.quantile(deltas, alpha)),
        "upper": float(np.quantile(deltas, 1 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
        "bootstrap_probability_second_gt_first": float(np.mean(deltas > 0)),
    }


def temporal_cross_validate(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_splits: int = 3,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    summary, _ = temporal_cross_validate_with_predictions(
        frame,
        features,
        n_splits=n_splits,
        bootstrap_iterations=bootstrap_iterations,
    )
    return summary


def temporal_cross_validate_with_predictions(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_splits: int = 3,
    bootstrap_iterations: int = 2_000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate expanding windows and return non-overlapping out-of-fold predictions."""
    fold_results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for temporal_fold in expanding_window_splits(frame, n_splits=n_splits):
        result = train_and_evaluate(temporal_fold.train, temporal_fold.test, features)
        fold_predictions = result.test_predictions.copy()
        fold_predictions["fold"] = temporal_fold.fold
        predictions.append(fold_predictions)
        fold_results.append(
            {
                "fold": temporal_fold.fold,
                "train_matches": len(temporal_fold.train),
                "test_matches": len(temporal_fold.test),
                "train_end_utc": pd.to_datetime(
                    temporal_fold.train[DEFAULT_TIME_COLUMN].max(), unit="ms", utc=True
                ).isoformat(),
                "test_start_utc": pd.to_datetime(
                    temporal_fold.test[DEFAULT_TIME_COLUMN].min(), unit="ms", utc=True
                ).isoformat(),
                "metrics": result.metrics,
            }
        )
    aucs = np.asarray([fold["metrics"]["roc_auc"] for fold in fold_results])
    out_of_fold = pd.concat(predictions, ignore_index=True).sort_values(
        [DEFAULT_TIME_COLUMN, "matchId"], kind="stable"
    )
    if out_of_fold["matchId"].duplicated().any():
        raise AssertionError("A match appears in more than one temporal test fold")
    pooled_interval = bootstrap_auc_interval(
        out_of_fold["win"].to_numpy(),
        out_of_fold["win_probability"].to_numpy(),
        iterations=bootstrap_iterations,
    )
    summary = {
        "n_splits": len(fold_results),
        "roc_auc_mean": float(aucs.mean()),
        "roc_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "pooled_test_matches": len(out_of_fold),
        "pooled_roc_auc_interval": pooled_interval,
        "folds": fold_results,
    }
    return summary, out_of_fold.reset_index(drop=True)
