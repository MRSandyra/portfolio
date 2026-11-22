from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Mapping
from uuid import uuid4

from . import __version__
from .config import RuntimeConfig


SAFE_CONFIG_FIELDS = (
    "max_news",
    "request_timeout",
    "retries",
    "backoff_seconds",
    "min_interval_seconds",
    "headless",
    "news_language",
    "cache_enabled",
    "cache_ttl_seconds",
    "log_level",
    "ticker_universe_max_age_days",
    "max_price_age_days",
    "max_fundamental_age_days",
    "extreme_return_threshold_percent",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(
    config: RuntimeConfig,
    artifacts: Mapping[str, Path],
) -> dict[str, object]:
    raw = asdict(config)
    safe_config = {field: raw[field] for field in SAFE_CONFIG_FIELDS}
    safe_config.update(
        {
            "driver_resolution": "explicit" if config.chrome_driver_path else "selenium_manager",
            "user_agent_configured": bool(config.user_agent.strip()),
            "sentiment_provider": config.sentiment_provider or "disabled",
        }
    )
    encoded = json.dumps(safe_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact_rows = []
    for role, path in sorted(artifacts.items()):
        resolved = path.resolve()
        artifact_rows.append(
            {
                "role": role,
                "filename": resolved.name,
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "indostock-analyzer",
            "version": __version__,
            "python": platform.python_version(),
        },
        "configuration": safe_config,
        "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        "input_artifacts": artifact_rows,
        "privacy": "Paths, environment variables, user-agent contents, and credentials are not recorded.",
    }


def minimal_run_metadata() -> dict[str, object]:
    return {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "indostock-analyzer",
            "version": __version__,
            "python": platform.python_version(),
        },
        "configuration": {},
        "configuration_sha256": None,
        "input_artifacts": [],
        "privacy": "Direct library invocation; runtime configuration lineage was not supplied.",
    }
