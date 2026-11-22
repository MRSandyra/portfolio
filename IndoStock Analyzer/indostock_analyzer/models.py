from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar


T = TypeVar("T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SourceAttribution:
    name: str
    url: str
    retrieved_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: SourceAttribution


@dataclass(frozen=True)
class FetchResult(Generic[T]):
    value: T | None
    source: SourceAttribution
    error: str | None = None
    error_code: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None


@dataclass(frozen=True)
class SentimentResult:
    polarity: float | None
    subjectivity: float | None
    label: str
    method: str
    language: str
    supported: bool
    note: str
    target: str | None = None
    confidence: float | None = None
    validated: bool = False
    validation_report_id: str | None = None
    evidence_level: str = "unverified"


@dataclass(frozen=True)
class ScoreResult:
    score: float | None
    label: str
    reasons: tuple[str, ...] = ()
    available_inputs: int = 0


@dataclass(frozen=True)
class CompositeIndicator:
    score: float | None
    label: str
    coverage: float
    components: dict[str, float | None]
    calibration: str = "not_backtested_or_calibrated"


def to_dict(value: Any) -> Any:
    """Convert nested dataclasses to JSON-safe primitives."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def source_from_dict(value: object) -> SourceAttribution:
    if not isinstance(value, dict):
        raise ValueError("cached source attribution must be an object")
    name = value.get("name")
    url = value.get("url")
    retrieved_at = value.get("retrieved_at")
    if not all(isinstance(item, str) and item for item in (name, url, retrieved_at)):
        raise ValueError("cached source attribution is incomplete")
    return SourceAttribution(name=name, url=url, retrieved_at=retrieved_at)
