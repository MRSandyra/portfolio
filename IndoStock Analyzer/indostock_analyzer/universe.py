from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


SYMBOL_PATTERN = re.compile(r"^[A-Z]{4}$")
SUPPORTED_STATUSES = frozenset({"listed", "suspended", "delisted"})
SUPPORTED_COVERAGE = frozenset({"complete", "curated_subset"})


class UniverseValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    name: str | None = None
    status: str = "listed"


@dataclass(frozen=True)
class TickerUniverse:
    market: str
    as_of: date
    coverage: str
    source_name: str
    source_url: str
    retrieved_at: datetime
    entries: tuple[UniverseEntry, ...]
    note: str | None = None

    @property
    def active_symbols(self) -> frozenset[str]:
        return frozenset(entry.symbol for entry in self.entries if entry.status != "delisted")

    def metadata(self, max_age_days: int, today: date | None = None) -> dict[str, Any]:
        reference_date = today or datetime.now(timezone.utc).date()
        age_days = (reference_date - self.as_of).days
        warnings: list[str] = []
        if age_days < 0:
            warnings.append("universe_as_of_date_is_in_the_future")
        if age_days > max_age_days:
            warnings.append("ticker_universe_snapshot_is_stale")
        if self.coverage != "complete":
            warnings.append("ticker_universe_is_a_curated_subset")
        official_source = _is_official_idx_url(self.source_url)
        if self.market.upper() == "IDX" and not official_source:
            warnings.append("ticker_universe_source_is_not_official_idx")
        return {
            "market": self.market,
            "as_of": self.as_of.isoformat(),
            "age_days": age_days,
            "coverage": self.coverage,
            "active_symbol_count": len(self.active_symbols),
            "official_source": official_source,
            "source": {
                "name": self.source_name,
                "url": self.source_url,
                "retrieved_at": self.retrieved_at.isoformat(),
            },
            "warnings": warnings,
            "note": self.note,
        }


def load_ticker_universe(path: Path) -> TickerUniverse:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UniverseValidationError(f"cannot read ticker universe: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UniverseValidationError(f"ticker universe is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise UniverseValidationError("ticker universe root must be an object")
    if payload.get("schema_version") != 1:
        raise UniverseValidationError("ticker universe schema_version must be 1")

    market = _required_string(payload, "market")
    coverage = _required_string(payload, "coverage")
    if coverage not in SUPPORTED_COVERAGE:
        raise UniverseValidationError("ticker universe coverage must be complete or curated_subset")
    try:
        as_of = date.fromisoformat(_required_string(payload, "as_of"))
    except ValueError as exc:
        raise UniverseValidationError("ticker universe as_of must use YYYY-MM-DD") from exc

    source = payload.get("source")
    if not isinstance(source, dict):
        raise UniverseValidationError("ticker universe source must be an object")
    source_name = _required_string(source, "name")
    source_url = _required_string(source, "url")
    if not source_url.startswith("https://"):
        raise UniverseValidationError("ticker universe source URL must use HTTPS")
    try:
        retrieved_at = datetime.fromisoformat(_required_string(source, "retrieved_at").replace("Z", "+00:00"))
    except ValueError as exc:
        raise UniverseValidationError("source retrieved_at must be an ISO-8601 timestamp") from exc
    if retrieved_at.tzinfo is None:
        raise UniverseValidationError("source retrieved_at must include a timezone")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise UniverseValidationError("ticker universe entries must be a non-empty list")
    entries: list[UniverseEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise UniverseValidationError(f"ticker entry {index} must be an object")
        symbol = _required_string(raw_entry, "symbol").upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise UniverseValidationError(f"ticker entry {index} has an invalid symbol")
        if symbol in seen:
            raise UniverseValidationError(f"duplicate ticker symbol: {symbol}")
        status = str(raw_entry.get("status", "listed")).strip().lower()
        if status not in SUPPORTED_STATUSES:
            raise UniverseValidationError(f"ticker {symbol} has an unsupported status")
        name_value = raw_entry.get("name")
        name = str(name_value).strip() if name_value is not None else None
        entries.append(UniverseEntry(symbol=symbol, name=name or None, status=status))
        seen.add(symbol)

    return TickerUniverse(
        market=market,
        as_of=as_of,
        coverage=coverage,
        source_name=source_name,
        source_url=source_url,
        retrieved_at=retrieved_at,
        entries=tuple(entries),
        note=str(payload.get("note", "")).strip() or None,
    )


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise UniverseValidationError(f"field {field} must be a non-empty string")
    return value.strip()


def read_universe_csv(path: Path) -> tuple[UniverseEntry, ...]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise UniverseValidationError(f"cannot read ticker CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise UniverseValidationError("ticker CSV has no header")
        normalized_headers = {_normalize_header(name): name for name in reader.fieldnames}
        symbol_header = next(
            (
                normalized_headers[key]
                for key in ("code", "kode", "stockcode", "kodesaham", "symbol")
                if key in normalized_headers
            ),
            None,
        )
        if symbol_header is None:
            raise UniverseValidationError("ticker CSV needs a Code/Kode/Stock Code/Symbol column")
        name_header = next(
            (
                normalized_headers[key]
                for key in ("companyname", "namaperusahaan", "name", "nama")
                if key in normalized_headers
            ),
            None,
        )
        status_header = normalized_headers.get("status")
        entries: list[UniverseEntry] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            symbol = str(row.get(symbol_header, "")).strip().upper()
            if not symbol:
                continue
            if not SYMBOL_PATTERN.fullmatch(symbol):
                raise UniverseValidationError(f"CSV row {row_number} has invalid symbol: {symbol}")
            if symbol in seen:
                raise UniverseValidationError(f"CSV contains duplicate symbol: {symbol}")
            status = str(row.get(status_header, "listed") if status_header else "listed").strip().lower()
            if status not in SUPPORTED_STATUSES:
                raise UniverseValidationError(f"CSV row {row_number} has unsupported status: {status}")
            name = str(row.get(name_header, "") if name_header else "").strip() or None
            entries.append(UniverseEntry(symbol=symbol, name=name, status=status))
            seen.add(symbol)
    if not entries:
        raise UniverseValidationError("ticker CSV contains no valid entries")
    return tuple(entries)


def write_ticker_universe(
    path: Path,
    entries: tuple[UniverseEntry, ...],
    as_of: date,
    source_name: str,
    source_url: str,
    coverage: str = "complete",
    note: str | None = None,
) -> None:
    if coverage not in SUPPORTED_COVERAGE:
        raise UniverseValidationError("coverage must be complete or curated_subset")
    if not source_url.startswith("https://"):
        raise UniverseValidationError("source URL must use HTTPS")
    payload = {
        "schema_version": 1,
        "market": "IDX",
        "as_of": as_of.isoformat(),
        "coverage": coverage,
        "source": {
            "name": source_name,
            "url": source_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        },
        "note": note,
        "entries": [
            {key: value for key, value in {
                "symbol": entry.symbol,
                "name": entry.name,
                "status": entry.status,
            }.items() if value is not None}
            for entry in entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_official_idx_url(value: str) -> bool:
    hostname = (urlparse(value).hostname or "").lower()
    return hostname == "idx.co.id" or hostname.endswith(".idx.co.id")
