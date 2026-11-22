from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .universe import TickerUniverse


DEFAULT_IDX_TICKERS = frozenset(
    {
        "ACES", "ADHI", "AKRA", "AMRT", "ARTO", "ASII", "AUTO", "AVIA",
        "BAPA", "BBCA", "BBNI", "BBRI", "BMRI", "BTPN", "COAL", "ESSA",
        "GOOD", "IPCC", "JPFA", "MAPI", "SDRA", "TLKM", "TOBA", "WSKT",
    }
)

TOKEN_PATTERN = re.compile(r"(?<![A-Z0-9])(?:\$)?([A-Z]{4})(?:\.JK)?(?![A-Z0-9])", re.IGNORECASE)


class TickerExtractor:
    def __init__(
        self,
        known_tickers: Iterable[str] = DEFAULT_IDX_TICKERS,
        universe: TickerUniverse | None = None,
        universe_max_age_days: int = 45,
    ):
        self.known_tickers = frozenset(ticker.upper() for ticker in known_tickers)
        self.universe = universe
        self.universe_max_age_days = universe_max_age_days

    @classmethod
    def from_universe(cls, universe: TickerUniverse, max_age_days: int = 45) -> "TickerExtractor":
        return cls(
            known_tickers=universe.active_symbols,
            universe=universe,
            universe_max_age_days=max_age_days,
        )

    def extract(self, text: str) -> list[str]:
        """Return unique known IDX tickers in first-seen order."""
        found: list[str] = []
        for match in TOKEN_PATTERN.finditer(text):
            ticker = match.group(1).upper()
            if ticker in self.known_tickers and ticker not in found:
                found.append(ticker)
        return found

    def metadata(self) -> dict[str, Any]:
        if self.universe is None:
            return {
                "market": "IDX",
                "coverage": "embedded_fallback_subset",
                "active_symbol_count": len(self.known_tickers),
                "warnings": ["ticker_universe_has_no_external_provenance"],
            }
        return self.universe.metadata(self.universe_max_age_days)
