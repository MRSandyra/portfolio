"""Uncertainty- and disagreement-based sampling for human annotation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tokopedia_sentiment.decision import predict_with_thresholds
from tokopedia_sentiment.triage import apply_triage_policy


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])


def create_active_learning_batch(
    frame: pd.DataFrame,
    model: Any,
    metadata: dict[str, Any],
    output_path: str | Path,
    *,
    batch_size: int = 500,
    exclude_review_ids: set[object] | None = None,
) -> tuple[Path, Path]:
    """Create a blind batch and a separate model/proxy key."""
    pool = frame.copy()
    if exclude_review_ids:
        pool = pool.loc[~pool["review_id"].isin(exclude_review_ids)].copy()
    if len(pool) < batch_size:
        raise ValueError("active-learning pool is smaller than requested batch")

    class_names = metadata["class_names"]
    probability = model.predict_proba(pool["text"].tolist())
    argmax = np.argmax(probability, axis=1)
    sorted_probability = np.sort(probability, axis=1)
    margin_uncertainty = 1.0 - (sorted_probability[:, -1] - sorted_probability[:, -2])
    entropy = _normalized_entropy(probability)

    policy = metadata.get("decision_policy", {})
    threshold_prediction = predict_with_thresholds(
        probability,
        class_names,
        policy.get("thresholds", {}),
        policy.get("majority_class", "positive"),
    )
    proxy_disagreement = argmax != pool["label"].to_numpy()
    threshold_disagreement = threshold_prediction != argmax
    triage_policy = metadata.get("triage_policy")
    triage_decision = (
        apply_triage_policy(probability, class_names, triage_policy)
        if triage_policy
        else np.full(len(pool), "not_available", dtype=object)
    )
    score = (
        0.45 * entropy
        + 0.35 * margin_uncertainty
        + 0.15 * proxy_disagreement.astype(float)
        + 0.05 * threshold_disagreement.astype(float)
    )

    scored = pool.copy()
    scored["uncertainty_score"] = score
    scored["argmax_label"] = [class_names[index] for index in argmax]
    scored["threshold_label"] = [class_names[index] for index in threshold_prediction]
    scored["triage_decision"] = triage_decision
    scored["proxy_disagreement"] = proxy_disagreement
    for index, name in enumerate(class_names):
        scored[f"probability_{name}"] = probability[:, index]

    scored = scored.sort_values("uncertainty_score", ascending=False)
    stratum_cap = max(5, math.ceil(batch_size / (len(class_names) * scored["category"].nunique())))
    diverse = scored.groupby(["argmax_label", "category"], group_keys=False).head(stratum_cap)
    selected = diverse.head(batch_size)
    if len(selected) < batch_size:
        remaining = scored.loc[~scored.index.isin(selected.index)]
        selected = pd.concat([selected, remaining.head(batch_size - len(selected))])
    selected = selected.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
    selected.insert(0, "annotation_id", range(1, len(selected) + 1))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    key_path = destination.with_name(f"{destination.stem}_key{destination.suffix}")
    blind = selected[["annotation_id", "review_id", "raw_text", "category"]].rename(
        columns={"raw_text": "review_text", "category": "product_category"}
    )
    blind["human_label"] = ""
    blind["annotator_id"] = ""
    blind["uncertain"] = ""
    blind["notes"] = ""
    blind.to_csv(destination, index=False)

    key_columns = [
        "annotation_id",
        "review_id",
        "rating",
        "label_name",
        "argmax_label",
        "threshold_label",
        "triage_decision",
        "proxy_disagreement",
        "uncertainty_score",
        *[f"probability_{name}" for name in class_names],
    ]
    selected[key_columns].rename(columns={"label_name": "proxy_label"}).to_csv(
        key_path, index=False
    )
    return destination, key_path
