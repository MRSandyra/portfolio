"""Three-class dataset preparation and leakage-aware split strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tokopedia_sentiment.constants import RANDOM_SEED
from tokopedia_sentiment.preprocessing import normalize_indonesian_text

CLASS_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class CaseStudyPreparationSummary:
    input_rows: int
    empty_text_rows_removed: int
    conflicting_text_rows_removed: int
    duplicate_rows_removed: int
    output_rows: int
    class_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseStudySplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str


def prepare_case_study_reviews(
    frame: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment_label",
    rating_column: str = "rating",
    date_column: str = "review_date",
    group_column: str = "product_id",
    category_column: str = "product_category",
    id_column: str = "review_id",
) -> tuple[pd.DataFrame, CaseStudyPreparationSummary]:
    """Validate and normalize the 2025 dataset while retaining audit metadata."""
    required = {
        text_column,
        label_column,
        rating_column,
        date_column,
        group_column,
        category_column,
        id_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    prepared = pd.DataFrame(
        {
            "review_id": frame[id_column],
            "raw_text": frame[text_column],
            "rating": pd.to_numeric(frame[rating_column], errors="raise").astype(int),
            "label_name": frame[label_column].astype(str).str.casefold().str.strip(),
            "date": pd.to_datetime(frame[date_column], errors="raise"),
            "group": frame[group_column],
            "category": frame[category_column].astype(str),
        }
    )
    unknown = sorted(set(prepared["label_name"]).difference(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"unknown sentiment labels: {', '.join(unknown)}")
    if not prepared["rating"].between(1, 5).all():
        raise ValueError("rating must be between 1 and 5")

    prepared["label"] = prepared["label_name"].map(LABEL_TO_ID).astype(int)
    prepared["text"] = prepared["raw_text"].map(normalize_indonesian_text)
    input_rows = len(prepared)

    empty_mask = prepared["text"].eq("")
    empty_count = int(empty_mask.sum())
    prepared = prepared.loc[~empty_mask].copy()

    label_counts = prepared.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    conflict_mask = prepared["text"].isin(conflicting_texts)
    conflict_count = int(conflict_mask.sum())
    prepared = prepared.loc[~conflict_mask].copy()

    before_dedup = len(prepared)
    prepared = prepared.sort_values(["date", "review_id"]).drop_duplicates("text", keep="last")
    duplicate_count = before_dedup - len(prepared)
    prepared = prepared.reset_index(drop=True)

    distribution = {name: int((prepared["label_name"] == name).sum()) for name in CLASS_NAMES}
    summary = CaseStudyPreparationSummary(
        input_rows=input_rows,
        empty_text_rows_removed=empty_count,
        conflicting_text_rows_removed=conflict_count,
        duplicate_rows_removed=duplicate_count,
        output_rows=len(prepared),
        class_distribution=distribution,
    )
    return prepared, summary


def _validate_split_classes(splits: CaseStudySplits) -> CaseStudySplits:
    expected = set(range(len(CLASS_NAMES)))
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        if set(frame["label"]) != expected:
            raise ValueError(f"{name} split does not contain every class")
    return splits


def temporal_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> CaseStudySplits:
    """Use older reviews for training and complete dates for later holdouts."""
    if frame["date"].isna().any():
        raise ValueError("temporal split requires valid dates")
    ordered = frame.sort_values(["date", "review_id"]).reset_index(drop=True)
    validation_position = round(len(ordered) * (1 - validation_size - test_size))
    test_position = round(len(ordered) * (1 - test_size))
    validation_cutoff = ordered.iloc[validation_position]["date"]
    test_cutoff = ordered.iloc[test_position]["date"]

    train = ordered.loc[ordered["date"] < validation_cutoff]
    validation = ordered.loc[
        (ordered["date"] >= validation_cutoff) & (ordered["date"] < test_cutoff)
    ]
    test = ordered.loc[ordered["date"] >= test_cutoff]
    return _validate_split_classes(
        CaseStudySplits(
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            test=test.reset_index(drop=True),
            strategy="temporal",
        )
    )


def group_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> CaseStudySplits:
    """Keep product IDs disjoint across train, validation, and test."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_validation_idx, test_idx = next(outer.split(frame, groups=frame["group"]))
    train_validation = frame.iloc[train_validation_idx]
    test = frame.iloc[test_idx]

    relative_validation = validation_size / (1 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=relative_validation, random_state=seed)
    train_idx, validation_idx = next(
        inner.split(train_validation, groups=train_validation["group"])
    )
    return _validate_split_classes(
        CaseStudySplits(
            train=train_validation.iloc[train_idx].reset_index(drop=True),
            validation=train_validation.iloc[validation_idx].reset_index(drop=True),
            test=test.reset_index(drop=True),
            strategy="product_group",
        )
    )


def random_stratified_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> CaseStudySplits:
    train_validation, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame["label"]
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_size / (1 - test_size),
        random_state=seed,
        stratify=train_validation["label"],
    )
    return _validate_split_classes(
        CaseStudySplits(
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            test=test.reset_index(drop=True),
            strategy="random_stratified",
        )
    )


def make_case_study_split(
    frame: pd.DataFrame,
    *,
    strategy: str,
    seed: int = RANDOM_SEED,
) -> CaseStudySplits:
    strategies = {
        "temporal": lambda: temporal_split(frame),
        "group": lambda: group_split(frame, seed=seed),
        "random": lambda: random_stratified_split(frame, seed=seed),
    }
    if strategy not in strategies:
        raise ValueError(f"unknown split strategy: {strategy}")
    return strategies[strategy]()


def split_audit(splits: CaseStudySplits) -> dict[str, Any]:
    """Report class, date, and product overlap characteristics."""
    frames = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    audit: dict[str, Any] = {"strategy": splits.strategy, "parts": {}}
    for name, frame in frames.items():
        audit["parts"][name] = {
            "rows": len(frame),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "groups": int(frame["group"].nunique()),
            "class_distribution": {
                class_name: int((frame["label_name"] == class_name).sum())
                for class_name in CLASS_NAMES
            },
        }
    audit["group_overlap"] = {
        "train_validation": len(set(splits.train["group"]) & set(splits.validation["group"])),
        "train_test": len(set(splits.train["group"]) & set(splits.test["group"])),
        "validation_test": len(set(splits.validation["group"]) & set(splits.test["group"])),
    }
    return audit
