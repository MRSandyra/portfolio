"""Static and tabular leakage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import FEATURE_REGISTRY, NON_MODEL_COLUMNS


@dataclass(frozen=True)
class LeakageAudit:
    cutoff_ms: int
    requested_features: tuple[str, ...]
    future_features: tuple[str, ...]
    unknown_features: tuple[str, ...]
    forbidden_columns: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.future_features or self.unknown_features or self.forbidden_columns)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def audit_feature_set(features: list[str], cutoff_ms: int) -> LeakageAudit:
    future = sorted(
        feature
        for feature in features
        if feature in FEATURE_REGISTRY and FEATURE_REGISTRY[feature].available_at_ms > cutoff_ms
    )
    forbidden = sorted(set(features).intersection(NON_MODEL_COLUMNS))
    unknown = sorted(set(features).difference(FEATURE_REGISTRY).difference(NON_MODEL_COLUMNS))
    return LeakageAudit(
        cutoff_ms=cutoff_ms,
        requested_features=tuple(features),
        future_features=tuple(future),
        unknown_features=tuple(unknown),
        forbidden_columns=tuple(forbidden),
    )


def assert_leakage_safe(frame: pd.DataFrame, features: list[str], cutoff_ms: int) -> LeakageAudit:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature columns missing from frame: {missing}")
    audit = audit_feature_set(features, cutoff_ms)
    if not audit.passed:
        raise ValueError(f"Leakage audit failed: {audit.to_dict()}")
    return audit

