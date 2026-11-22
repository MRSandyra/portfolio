"""Static visualizations whose interpretation is documented in the summary."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flight-analysis-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_airline_availability(airlines: pd.DataFrame, path: Path) -> None:
    data = airlines.head(10).sort_values("offer_count")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(data["airline"], data["offer_count"], color="#315A7D")
    ax.set_title("Flight offer availability by airline (not sales)", loc="left", weight="bold")
    ax.set_xlabel("Unique valid offer rows")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_fare_by_stops(stops: pd.DataFrame, path: Path) -> None:
    data = stops.sort_values("stop_count")
    lower = data["median_fare_usd"] - data["fare_q1_usd"]
    upper = data["fare_q3_usd"] - data["median_fare_usd"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        data["stop_count"],
        data["median_fare_usd"],
        yerr=[lower, upper],
        fmt="o",
        capsize=5,
        color="#C4553D",
    )
    ax.set_title("Listed fare by stop count", loc="left", weight="bold")
    ax.set_xlabel("Number of stops")
    ax.set_ylabel("Median fare (USD); whiskers show IQR")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_top_route_fares(routes: pd.DataFrame, path: Path) -> None:
    data = routes.head(10).sort_values("median_fare_usd")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(data["route"], data["median_fare_usd"], color="#638B66")
    ax.set_title("Median fare for most-observed routes", loc="left", weight="bold")
    ax.set_xlabel("Median listed fare (USD)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def create_charts(
    airlines: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = [
        output_dir / "airline_offer_availability.png",
        output_dir / "fare_by_stop_count.png",
    ]
    plot_airline_availability(airlines, paths[0])
    plot_fare_by_stops(stops, paths[1])
    if not routes.empty:
        route_path = output_dir / "top_route_fares.png"
        plot_top_route_fares(routes, route_path)
        paths.append(route_path)
    return paths

