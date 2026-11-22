"""Static Matplotlib figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .lorenz import lorenz_from_group
from .simulation import SimulationResult

COUNTRY_NAMES = {
    "BRA": "Brazil",
    "CHN": "China",
    "IDN": "Indonesia",
    "USA": "United States",
    "ZAF": "South Africa",
}


def _save(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(figure)


def plot_gini_trends(gini: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for country, subset in gini.groupby("country_code", sort=True):
        axis.plot(
            subset["year"],
            subset["gini"],
            marker="o",
            linewidth=2,
            label=COUNTRY_NAMES.get(country, country),
        )
    axis.set(
        title="Grouped-data Gini in the committed PIP sample",
        xlabel="Survey year (not harmonized across countries)",
        ylabel="Gini coefficient",
        ylim=(0, 0.75),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    _save(figure, destination)


def plot_lorenz(group: pd.DataFrame, destination: Path) -> None:
    population, welfare = lorenz_from_group(group)
    country = group["country_code"].iloc[0]
    year = int(group["year"].iloc[0])
    welfare_type = group["welfare_type"].iloc[0]
    figure, axis = plt.subplots(figsize=(6.2, 6.2))
    axis.plot(population, welfare, linewidth=2.5, label="Observed grouped distribution")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Equality line")
    axis.set(
        title=f"Lorenz curve: {COUNTRY_NAMES.get(country, country)}, {year}",
        xlabel="Cumulative population share",
        ylabel=f"Cumulative {welfare_type} share",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, destination)


def plot_simulations(
    original: pd.DataFrame,
    simulations: list[SimulationResult],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6.2))
    population, welfare = lorenz_from_group(original)
    axis.plot(population, welfare, linewidth=2.5, label="Observed")
    for simulation in simulations:
        sim_population, sim_welfare = lorenz_from_group(simulation.distribution)
        axis.plot(sim_population, sim_welfare, linewidth=2, label=simulation.name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Equality line")
    axis.set(
        title="Illustrative simulations — not policy forecasts",
        xlabel="Cumulative population share",
        ylabel="Cumulative welfare share",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save(figure, destination)

