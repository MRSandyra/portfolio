"""Create privacy-conscious, stratified human-annotation batches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def annotation_key_path(output_path: str | Path) -> Path:
    destination = Path(output_path)
    return destination.with_name(f"{destination.stem}_key{destination.suffix}")


def create_annotation_batch(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    samples_per_rating: int = 100,
    seed: int = 42,
) -> Path:
    """Sample review text by rating and leave human labels intentionally blank."""
    required = {"review_id", "review_text", "rating", "sentiment_label", "product_category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    samples = []
    for rating, group in frame.groupby("rating", sort=True):
        count = min(samples_per_rating, len(group))
        sampled = group.sample(n=count, random_state=seed + int(rating)).copy()
        samples.append(sampled)
    batch = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed)
    batch = batch[
        ["review_id", "review_text", "rating", "sentiment_label", "product_category"]
    ].rename(columns={"sentiment_label": "proxy_label"})
    batch.insert(0, "annotation_id", range(1, len(batch) + 1))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    batch[["annotation_id", "rating", "proxy_label"]].to_csv(
        annotation_key_path(destination), index=False
    )
    blind = batch[["annotation_id", "review_id", "review_text", "product_category"]].copy()
    blind["human_label"] = ""
    blind["annotator_id"] = ""
    blind["uncertain"] = ""
    blind["notes"] = ""
    blind.to_csv(destination, index=False)
    return destination
