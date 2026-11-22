"""Transparent illustrative redistribution simulations (not forecasts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lorenz import gini_from_group


@dataclass(frozen=True)
class SimulationResult:
    name: str
    description: str
    distribution: pd.DataFrame
    gini: float


def _with_simulated_shares(group: pd.DataFrame, simulated_welfare: pd.Series) -> pd.DataFrame:
    if not np.isfinite(simulated_welfare).all() or (simulated_welfare < 0).any():
        raise ValueError("simulated welfare must be finite and non-negative")
    result = group.copy()
    totals = simulated_welfare * result["pop"]
    if totals.sum() <= 0:
        raise ValueError("simulated total welfare must be positive")
    result["avg_welfare"] = simulated_welfare
    result["welfare_share"] = totals / totals.sum()
    return result


def simulate_bottom_subsidy(
    group: pd.DataFrame,
    *,
    amount_per_person_per_day: float = 0.5,
    bottom_percentile: int = 50,
) -> SimulationResult:
    """Add an externally funded fixed subsidy to the selected bottom bins."""

    if amount_per_person_per_day < 0 or not 1 <= bottom_percentile <= 100:
        raise ValueError("invalid subsidy amount or percentile cutoff")
    simulated = group["avg_welfare"].copy()
    simulated.loc[group["percentile"] <= bottom_percentile] += amount_per_person_per_day
    distribution = _with_simulated_shares(group, simulated)
    return SimulationResult(
        name="Illustrative bottom-50 subsidy",
        description=(
            f"Adds {amount_per_person_per_day:g} 2021 PPP dollars per person per day "
            f"to percentiles 1-{bottom_percentile}; external funding is assumed."
        ),
        distribution=distribution,
        gini=gini_from_group(distribution),
    )


def simulate_progressive_tax(
    group: pd.DataFrame,
    *,
    tax_rate: float = 0.10,
    taxed_above_percentile: int = 90,
    subsidized_through_percentile: int = 40,
) -> SimulationResult:
    """Apply a static tax and transfer all revenue uniformly to bottom bins."""

    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    if not 1 <= subsidized_through_percentile < taxed_above_percentile <= 99:
        raise ValueError("percentile cutoffs must define non-overlapping groups")

    taxed = group["percentile"] > taxed_above_percentile
    subsidized = group["percentile"] <= subsidized_through_percentile
    if not taxed.any() or not subsidized.any():
        raise ValueError("selected tax and subsidy groups must be non-empty")

    simulated = group["avg_welfare"].copy()
    revenue = (simulated.loc[taxed] * tax_rate * group.loc[taxed, "pop"]).sum()
    recipient_population = group.loc[subsidized, "pop"].sum()
    simulated.loc[taxed] *= 1.0 - tax_rate
    simulated.loc[subsidized] += revenue / recipient_population
    distribution = _with_simulated_shares(group, simulated)
    return SimulationResult(
        name="Illustrative top-10 tax / bottom-40 transfer",
        description=(
            f"Applies a {tax_rate:.0%} static tax above percentile "
            f"{taxed_above_percentile} and transfers all arithmetic revenue per capita "
            f"through percentile {subsidized_through_percentile}."
        ),
        distribution=distribution,
        gini=gini_from_group(distribution),
    )

