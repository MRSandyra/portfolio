"""Download, select, and validate World Bank PIP percentile data."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

PIP_VERSION = "20260922_2021"
PIP_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0063646/"
    "DR0090357/world_100bin.csv"
)
PIP_SHA256 = "6d35e14b8cbbb3effe53d52127d874d593eb0960810841368a9f6e83ce3943d1"

KEY_COLUMNS = ["country_code", "year", "reporting_level", "welfare_type"]
NUMERIC_COLUMNS = [
    "year",
    "percentile",
    "avg_welfare",
    "pop_share",
    "welfare_share",
    "quantile",
    "pop",
]
INPUT_COLUMNS = KEY_COLUMNS + [
    "percentile",
    "avg_welfare",
    "pop_share",
    "welfare_share",
    "quantile",
    "pop",
]
REQUIRED_VALUES = KEY_COLUMNS + [
    "percentile",
    "avg_welfare",
    "pop_share",
    "welfare_share",
    "pop",
]

SAMPLE_SELECTION: tuple[tuple[str, int], ...] = (
    ("BRA", 2001),
    ("BRA", 2011),
    ("BRA", 2024),
    ("CHN", 1999),
    ("CHN", 2010),
    ("CHN", 2022),
    ("IDN", 2000),
    ("IDN", 2010),
    ("IDN", 2019),
    ("IDN", 2025),
    ("USA", 2000),
    ("USA", 2010),
    ("USA", 2024),
    ("ZAF", 2000),
    ("ZAF", 2010),
    ("ZAF", 2022),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the pinned PIP CSV and reject a checksum mismatch."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if sha256(destination) != PIP_SHA256:
            raise ValueError(f"existing file has an unexpected checksum: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(PIP_URL, temporary)
        actual = sha256(temporary)
        if actual != PIP_SHA256:
            raise ValueError(f"checksum mismatch: expected {PIP_SHA256}, got {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_percentiles(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_percentiles(
    data: pd.DataFrame,
    *,
    reporting_level: str = "national",
    countries: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    require_100_bins: bool = True,
) -> pd.DataFrame:
    """Select and fail-fast validate complete PIP distributions.

    Missing values are not imputed and individual percentile rows are never
    silently dropped, because either action would change distribution shares.
    """

    missing_columns = set(INPUT_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(f"missing columns: {', '.join(sorted(missing_columns))}")

    prepared = data.loc[data["reporting_level"].eq(reporting_level)].copy()
    if countries is not None:
        wanted_countries = {country.upper() for country in countries}
        prepared = prepared.loc[prepared["country_code"].isin(wanted_countries)]
    if years is not None:
        prepared = prepared.loc[prepared["year"].isin(set(years))]
    if prepared.empty:
        raise ValueError("selection produced no rows")

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[REQUIRED_VALUES].isna().any().any():
        counts = prepared[REQUIRED_VALUES].isna().sum()
        details = ", ".join(f"{name}={int(count)}" for name, count in counts.items() if count)
        raise ValueError(f"missing or non-numeric required values: {details}")
    if not np.isfinite(prepared[REQUIRED_VALUES].select_dtypes(include="number")).all().all():
        raise ValueError("required numeric values must be finite")

    if (prepared["pop_share"] <= 0).any():
        raise ValueError("pop_share must be strictly positive")
    if (prepared["welfare_share"] < 0).any():
        raise ValueError("negative welfare_share is outside this analysis scope")
    if (prepared["avg_welfare"] < 0).any() or (prepared["pop"] <= 0).any():
        raise ValueError("avg_welfare must be non-negative and pop must be positive")

    group_columns = KEY_COLUMNS
    duplicates = prepared.duplicated(group_columns + ["percentile"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate percentile rows found within a distribution")

    if require_100_bins:
        expected = set(range(1, 101))
        for key, group in prepared.groupby(group_columns, sort=False, dropna=False):
            actual = set(group["percentile"].astype(int))
            if actual != expected or len(group) != 100:
                raise ValueError(f"incomplete percentile distribution for {key}")
            if not np.isclose(group["pop_share"].sum(), 1.0, atol=1e-6):
                raise ValueError(f"pop_share does not sum to one for {key}")
            if not np.isclose(group["welfare_share"].sum(), 1.0, atol=1e-6):
                raise ValueError(f"welfare_share does not sum to one for {key}")
            ordered_welfare = group.sort_values("percentile")["avg_welfare"].to_numpy()
            if (np.diff(ordered_welfare) < -1e-10).any():
                raise ValueError(f"avg_welfare is not ordered for {key}")

    return prepared.sort_values(group_columns + ["percentile"], kind="stable").reset_index(
        drop=True
    )


def create_sample(source: Path, destination: Path) -> Path:
    """Create the documented, deterministic country-year sample."""

    data = prepare_percentiles(load_percentiles(source), reporting_level="national")
    selection = pd.MultiIndex.from_tuples(SAMPLE_SELECTION, names=["country_code", "year"])
    indexed = data.set_index(["country_code", "year"])
    sample = indexed.loc[indexed.index.isin(selection)].reset_index()
    found = set(map(tuple, sample[["country_code", "year"]].drop_duplicates().to_numpy()))
    if found != set(SAMPLE_SELECTION):
        missing = sorted(set(SAMPLE_SELECTION).difference(found))
        raise ValueError(f"source does not contain the complete sample selection: {missing}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False, lineterminator="\n", float_format="%.15g")
    return destination

