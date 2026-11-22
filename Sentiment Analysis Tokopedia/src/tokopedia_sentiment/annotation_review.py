"""Agreement analysis, disagreement queues, and adjudicated gold-set assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID_LABELS = {"negative", "neutral", "positive"}


def _read_annotator(path: str | Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"annotation_id", "review_id", "review_text", "human_label", "uncertain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    labels = frame["human_label"].astype(str).str.casefold().str.strip()
    uncertain = frame["uncertain"].astype(str).str.casefold().isin({"true", "1", "yes", "ya"})
    invalid = set(labels.loc[~uncertain & labels.ne("")]).difference(VALID_LABELS)
    if invalid:
        raise ValueError(f"{path} contains invalid labels: {', '.join(sorted(invalid))}")
    frame = frame.copy()
    frame[f"label_{suffix}"] = labels.mask(uncertain, "uncertain")
    return frame[["annotation_id", "review_id", "review_text", f"label_{suffix}"]].drop_duplicates(
        "annotation_id"
    )


def analyze_annotation_agreement(
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    *,
    report_path: str | Path,
    disagreement_path: str | Path,
) -> dict[str, Any]:
    first = _read_annotator(annotator_a_path, "a")
    second = _read_annotator(annotator_b_path, "b")
    merged = first.merge(
        second,
        on=["annotation_id", "review_id", "review_text"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("annotator files do not contain the same annotation rows")
    complete = merged.loc[
        merged["label_a"].isin(VALID_LABELS) & merged["label_b"].isin(VALID_LABELS)
    ].copy()
    if complete.empty:
        raise ValueError("no rows contain two completed valid labels")
    agreement = complete["label_a"].eq(complete["label_b"])
    disagreements = complete.loc[~agreement].copy()
    disagreements["adjudicated_label"] = ""
    destination = Path(disagreement_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(destination, index=False)
    report = {
        "annotator_rows": len(merged),
        "complete_pairs": len(complete),
        "incomplete_or_uncertain_rows": len(merged) - len(complete),
        "agreements": int(agreement.sum()),
        "disagreements": len(disagreements),
        "raw_agreement": float(agreement.mean()),
        "cohen_kappa": float(cohen_kappa_score(complete["label_a"], complete["label_b"])),
        "disagreement_file": str(destination),
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_adjudicated_gold_set(
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    adjudication_path: str | Path,
    output_path: str | Path,
) -> Path:
    first = _read_annotator(annotator_a_path, "a")
    second = _read_annotator(annotator_b_path, "b")
    merged = first.merge(
        second,
        on=["annotation_id", "review_id", "review_text"],
        validate="one_to_one",
    )
    adjudication = pd.read_csv(adjudication_path, keep_default_na=False)
    required = {"annotation_id", "adjudicated_label"}
    if not required.issubset(adjudication.columns):
        raise ValueError("adjudication file must contain annotation_id and adjudicated_label")
    adjudication["adjudicated_label"] = (
        adjudication["adjudicated_label"].astype(str).str.casefold().str.strip()
    )
    invalid = set(adjudication["adjudicated_label"]).difference(VALID_LABELS | {""})
    if invalid:
        raise ValueError(f"invalid adjudicated labels: {', '.join(sorted(invalid))}")
    adjudicated = adjudication.set_index("annotation_id")["adjudicated_label"].to_dict()

    labels = []
    sources = []
    for row in merged.itertuples(index=False):
        if row.label_a in VALID_LABELS and row.label_a == row.label_b:
            labels.append(row.label_a)
            sources.append("annotator_agreement")
        else:
            label = adjudicated.get(row.annotation_id, "")
            labels.append(label)
            sources.append("adjudicated" if label else "unresolved")
    merged["human_label"] = labels
    merged["label_source"] = sources
    gold = merged.loc[merged["human_label"].isin(VALID_LABELS)].copy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gold[["annotation_id", "review_id", "review_text", "human_label", "label_source"]].to_csv(
        destination, index=False
    )
    return destination
