from datetime import date
import json
from pathlib import Path

import pytest

from indostock_analyzer.tickers import TickerExtractor
from indostock_analyzer.universe import (
    UniverseValidationError,
    load_ticker_universe,
    read_universe_csv,
    write_ticker_universe,
)


def test_default_snapshot_has_provenance_and_expands_ticker_coverage():
    universe = load_ticker_universe(Path("config/idx_tickers.json"))
    metadata = universe.metadata(max_age_days=45, today=date(2026, 11, 22))

    assert len(universe.active_symbols) == 43
    assert metadata["official_source"] is True
    assert metadata["coverage"] == "curated_subset"
    assert metadata["age_days"] == 0
    assert "ticker_universe_is_a_curated_subset" in metadata["warnings"]
    assert TickerExtractor.from_universe(universe).extract("AADI dan BBRI") == ["AADI", "BBRI"]


def test_delisted_symbols_are_excluded_from_active_extraction(tmp_path):
    payload = {
        "schema_version": 1,
        "market": "IDX",
        "as_of": "2026-11-22",
        "coverage": "complete",
        "source": {
            "name": "IDX",
            "url": "https://www.idx.co.id/example",
            "retrieved_at": "2026-11-22T00:00:00+07:00",
        },
        "entries": [
            {"symbol": "BBRI", "status": "listed"},
            {"symbol": "SRIL", "status": "delisted"},
        ],
    }
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    universe = load_ticker_universe(path)

    assert universe.active_symbols == frozenset({"BBRI"})
    assert TickerExtractor.from_universe(universe).extract("BBRI dan SRIL") == ["BBRI"]


def test_duplicate_symbols_are_rejected(tmp_path):
    payload = {
        "schema_version": 1,
        "market": "IDX",
        "as_of": "2026-11-22",
        "coverage": "complete",
        "source": {
            "name": "IDX",
            "url": "https://www.idx.co.id/example",
            "retrieved_at": "2026-11-22T00:00:00+07:00",
        },
        "entries": [{"symbol": "BBRI"}, {"symbol": "BBRI"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UniverseValidationError, match="duplicate"):
        load_ticker_universe(path)


def test_official_csv_can_be_imported_and_revalidated(tmp_path):
    csv_path = tmp_path / "idx.csv"
    csv_path.write_text(
        "Kode,Nama Perusahaan,Status\nBBRI,Bank Rakyat Indonesia,listed\nTLKM,Telkom Indonesia,listed\n",
        encoding="utf-8",
    )
    entries = read_universe_csv(csv_path)
    output = tmp_path / "universe.json"
    write_ticker_universe(
        output,
        entries,
        as_of=date(2026, 11, 22),
        source_name="Indonesia Stock Exchange",
        source_url="https://www.idx.co.id/en/listed-companies/company-profiles/",
    )

    loaded = load_ticker_universe(output)
    assert loaded.active_symbols == frozenset({"BBRI", "TLKM"})
    assert loaded.coverage == "complete"
