"""Dataset drift diagnostics that do not require model retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _distribution(series: pd.Series, categories: list[object]) -> np.ndarray:
    counts = series.value_counts(normalize=True)
    return np.asarray([float(counts.get(category, 0.0)) for category in categories])


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = np.clip(first, 1e-12, 1.0)
    second = np.clip(second, 1e-12, 1.0)
    first = first / first.sum()
    second = second / second.sum()
    midpoint = 0.5 * (first + second)
    divergence = 0.5 * np.sum(first * np.log(first / midpoint))
    divergence += 0.5 * np.sum(second * np.log(second / midpoint))
    return float(divergence)


def _population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    bins = np.asarray([0, 4, 8, 16, 32, 64, np.inf])
    reference_hist = np.histogram(reference, bins=bins)[0] / len(reference)
    current_hist = np.histogram(current, bins=bins)[0] / len(current)
    reference_hist = np.clip(reference_hist, 1e-6, 1.0)
    current_hist = np.clip(current_hist, 1e-6, 1.0)
    return float(np.sum((current_hist - reference_hist) * np.log(current_hist / reference_hist)))


def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Compare prevalence, category, rating, length, and vocabulary coverage."""
    categorical: dict[str, Any] = {}
    for column in ("label_name", "category", "rating"):
        categories = sorted(set(reference[column]).union(current[column]), key=str)
        reference_distribution = _distribution(reference[column], categories)
        current_distribution = _distribution(current[column], categories)
        categorical[column] = {
            "jensen_shannon_divergence": _jensen_shannon(
                reference_distribution, current_distribution
            ),
            "reference": {
                str(category): float(value)
                for category, value in zip(categories, reference_distribution, strict=True)
            },
            "current": {
                str(category): float(value)
                for category, value in zip(categories, current_distribution, strict=True)
            },
        }

    reference_length = reference["text"].str.split().str.len().to_numpy()
    current_length = current["text"].str.split().str.len().to_numpy()
    vocabulary = set(" ".join(reference["text"]).split())
    current_tokens = " ".join(current["text"]).split()
    oov_count = sum(token not in vocabulary for token in current_tokens)
    oov_rate = float(oov_count / len(current_tokens)) if current_tokens else 0.0
    report = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_date_range": [
            reference["date"].min().date().isoformat(),
            reference["date"].max().date().isoformat(),
        ],
        "current_date_range": [
            current["date"].min().date().isoformat(),
            current["date"].max().date().isoformat(),
        ],
        "categorical": categorical,
        "text": {
            "reference_mean_tokens": float(reference_length.mean()),
            "current_mean_tokens": float(current_length.mean()),
            "token_length_psi": _population_stability_index(reference_length, current_length),
            "current_oov_token_rate": oov_rate,
        },
    }
    report["alerts"] = {
        "label_shift": categorical["label_name"]["jensen_shannon_divergence"] > 0.10,
        "category_shift": categorical["category"]["jensen_shannon_divergence"] > 0.10,
        "rating_shift": categorical["rating"]["jensen_shannon_divergence"] > 0.10,
        "token_length_shift": report["text"]["token_length_psi"] > 0.20,
        "high_oov_rate": oov_rate > 0.10,
    }
    return report


def write_drift_report(report: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return destination
