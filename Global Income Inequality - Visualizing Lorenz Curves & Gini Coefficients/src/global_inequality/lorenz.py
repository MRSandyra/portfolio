"""Lorenz-curve and grouped-data Gini calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _validated_shares(
    population_share: Iterable[float], welfare_share: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    population = np.asarray(list(population_share), dtype=float)
    welfare = np.asarray(list(welfare_share), dtype=float)

    if population.ndim != 1 or welfare.ndim != 1:
        raise ValueError("population_share and welfare_share must be one-dimensional")
    if population.size < 2 or population.size != welfare.size:
        raise ValueError("share arrays must have the same length and at least two bins")
    if not np.isfinite(population).all() or not np.isfinite(welfare).all():
        raise ValueError("shares must be finite")
    if (population <= 0).any():
        raise ValueError("population shares must be strictly positive")
    if (welfare < 0).any():
        raise ValueError("classical Lorenz curves require non-negative welfare shares")

    population_total = population.sum()
    welfare_total = welfare.sum()
    if population_total <= 0 or welfare_total <= 0:
        raise ValueError("share totals must be positive")

    return population / population_total, welfare / welfare_total


def lorenz_curve(
    population_share: Iterable[float], welfare_share: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative population and welfare shares, including (0, 0).

    Inputs must already be ordered from the lowest to the highest welfare bin.
    Shares are normalized by their totals so the result is scale invariant.
    """

    population, welfare = _validated_shares(population_share, welfare_share)
    cumulative_population = np.concatenate(([0.0], np.cumsum(population)))
    cumulative_welfare = np.concatenate(([0.0], np.cumsum(welfare)))
    cumulative_population[-1] = 1.0
    cumulative_welfare[-1] = 1.0
    return cumulative_population, cumulative_welfare


def gini_coefficient(
    population_share: Iterable[float], welfare_share: Iterable[float]
) -> float:
    """Compute the grouped-data Gini as one minus twice the Lorenz area."""

    cumulative_population, cumulative_welfare = lorenz_curve(
        population_share, welfare_share
    )
    area = np.trapezoid(cumulative_welfare, cumulative_population)
    gini = float(1.0 - 2.0 * area)
    if gini < -1e-12 or gini > 1.0 + 1e-12:
        raise ValueError("ordered shares imply a Gini outside [0, 1]")
    return float(np.clip(gini, 0.0, 1.0))


def _ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    required = {"percentile", "pop_share", "welfare_share"}
    missing = required.difference(group.columns)
    if missing:
        raise ValueError(f"group is missing columns: {', '.join(sorted(missing))}")
    if group["percentile"].duplicated().any():
        raise ValueError("percentile values must be unique within a distribution")
    return group.sort_values("percentile", kind="stable")


def lorenz_from_group(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Lorenz curve from a PIP-style percentile group."""

    ordered = _ordered_group(group)
    return lorenz_curve(ordered["pop_share"], ordered["welfare_share"])


def gini_from_group(group: pd.DataFrame) -> float:
    """Compute a grouped-data Gini from a PIP-style percentile group."""

    ordered = _ordered_group(group)
    return gini_coefficient(ordered["pop_share"], ordered["welfare_share"])

