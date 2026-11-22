from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from uuid import uuid4


LOGGER = logging.getLogger(__name__)
SAFE_NAMESPACE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class CacheEntry:
    value: Any
    stored_at: str
    age_seconds: float


class JsonTtlCache:
    """Small, atomic JSON cache for public source responses."""

    def __init__(
        self,
        root: Path,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ):
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
        self.root = root
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock

    def get(self, namespace: str, key: str) -> CacheEntry | None:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stored_at_epoch = float(payload["stored_at_epoch"])
            value = payload["value"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "cache_read_failed",
                extra={"event": "cache_read_failed", "context": {"path": str(path), "error_type": type(exc).__name__}},
            )
            self._discard(path)
            return None

        age = max(0.0, self._clock() - stored_at_epoch)
        if age > self.ttl_seconds:
            self._discard(path)
            return None
        stored_at = datetime.fromtimestamp(stored_at_epoch, timezone.utc).isoformat()
        return CacheEntry(value=value, stored_at=stored_at, age_seconds=age)

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stored_at_epoch": self._clock(), "value": value}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _path(self, namespace: str, key: str) -> Path:
        safe_namespace = SAFE_NAMESPACE.sub("_", namespace).strip("._") or "default"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / safe_namespace / f"{digest}.json"

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning(
                "cache_discard_failed",
                extra={"event": "cache_discard_failed", "context": {"path": str(path)}},
            )
