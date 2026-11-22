"""Dataset validation, labeling, provenance, and leakage-safe splitting."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tokopedia_sentiment.constants import RANDOM_SEED
from tokopedia_sentiment.preprocessing import normalize_indonesian_text


@dataclass(frozen=True)
class PreparationSummary:
    input_rows: int
    empty_text_rows_removed: int
    conflicting_text_rows_removed: int
    duplicate_rows_removed: int
    output_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 fingerprint without loading the dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Load a local CSV. The raw dataset is intentionally not bundled in this repo."""
    return pd.read_csv(path)


def rating_to_label(rating: int | float) -> int:
    """Map ratings 1-3 to negative (0), and 4-5 to positive (1)."""
    numeric = float(rating)
    if numeric < 1 or numeric > 5 or not numeric.is_integer():
        raise ValueError(f"rating must be an integer from 1 to 5; got {rating!r}")
    return 0 if numeric <= 3 else 1


def prepare_reviews(
    frame: pd.DataFrame,
    *,
    text_column: str = "Review",
    rating_column: str = "Rating",
) -> tuple[pd.DataFrame, PreparationSummary]:
    """Validate, label, normalize, and deduplicate reviews before splitting.

    A normalized text appearing with both labels is removed entirely. This avoids the
    same wording crossing splits with contradictory labels. Exact normalized duplicates
    keep the first row only.
    """
    missing = {text_column, rating_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    prepared = frame[[text_column, rating_column]].copy()
    prepared.columns = ["raw_text", "rating"]
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="raise")
    prepared["label"] = prepared["rating"].map(rating_to_label)
    prepared["text"] = prepared["raw_text"].map(normalize_indonesian_text)

    input_rows = len(prepared)
    empty_mask = prepared["text"].eq("")
    empty_count = int(empty_mask.sum())
    prepared = prepared.loc[~empty_mask].copy()

    conflicting_texts = prepared.groupby("text")["label"].nunique()
    conflicting_texts = conflicting_texts[conflicting_texts > 1].index
    conflict_mask = prepared["text"].isin(conflicting_texts)
    conflict_count = int(conflict_mask.sum())
    prepared = prepared.loc[~conflict_mask].copy()

    before_dedup = len(prepared)
    prepared = prepared.drop_duplicates(subset="text", keep="first")
    duplicate_count = before_dedup - len(prepared)
    prepared = prepared[["text", "label", "rating"]].reset_index(drop=True)

    summary = PreparationSummary(
        input_rows=input_rows,
        empty_text_rows_removed=empty_count,
        conflicting_text_rows_removed=conflict_count,
        duplicate_rows_removed=duplicate_count,
        output_rows=len(prepared),
    )
    return prepared, summary


def stratified_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> DatasetSplits:
    """Create mutually exclusive stratified splits from an already deduplicated frame."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to < 1")
    if "label" not in frame:
        raise ValueError("frame must contain a label column")
    if frame["label"].nunique() != 2:
        raise ValueError("both sentiment classes are required")

    # Pass integer counts to sklearn so binary floating-point rounding cannot turn a
    # requested 15% into 16 rows during the second split.
    test_count = max(2, round(len(frame) * test_size))
    validation_count = max(2, round(len(frame) * validation_size))
    if test_count + validation_count >= len(frame):
        raise ValueError("dataset is too small for the requested three-way split")

    train_validation, test = train_test_split(
        frame,
        test_size=test_count,
        random_state=seed,
        stratify=frame["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_count,
        random_state=seed,
        stratify=train_validation["label"],
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
