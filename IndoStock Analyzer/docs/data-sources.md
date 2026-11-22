# Data source policy

## Current sources

| Purpose | Adapter | Strength | Material limitation |
|---|---|---|---|
| Ticker universe | Local, versioned snapshot with IDX provenance | Reproducible symbols, status, source, and `as_of` date | Bootstrap snapshot is a curated subset until replaced with a complete official export |
| News discovery | IndoPremier browser adapter | Preserves headline URL and retrieval time | Markup and browser behavior can change |
| Market data | Yahoo Finance adapter | Covers fundamental fields and daily history through one interface | Field availability, units, delays, and revisions are controlled upstream |

The pipeline records failures instead of substituting values. Cached values retain the original retrieval timestamp and are used only within the configured TTL.

## Universe update workflow

The authoritative reference is the IDX [Company Profiles](https://www.idx.co.id/en/listed-companies/company-profiles/) or [Stock List](https://www.idx.co.id/en/market-data/stocks-data/stock-list/) page. Automated access can be blocked by the site's protection layer, so the application does not bypass it or depend on an undocumented endpoint.

Download an official CSV through the permitted user flow, import it with `indostock-universe import-csv`, and review the generated `as_of`, `coverage`, and source metadata. Keep delisted entries with `status: delisted` when historical provenance matters; the extractor ignores them for current news.

## Criteria for another provider

A replacement or fallback provider should be added only when all of these are documented:

1. Permitted automated access and attribution requirements.
2. Stable identifiers for IDX securities and corporate actions.
3. Explicit field definitions, currencies, units, adjustment policy, and timestamps.
4. Rate limits, timeout behavior, retry guidance, and service availability.
5. Historical coverage needed for validation without look-ahead or survivorship bias.
6. A test fixture or sandbox that CI can use without credentials.

## Fallback boundary

Automatic mixing of providers is intentionally absent. A fallback can change definitions or adjustment policy while appearing to fill a missing field. A future provider adapter should first expose its source and schema separately. Cross-provider normalization and comparison tests must pass before a fallback contributes to an indicator.

Official exchange or licensed provider data is preferable when its access terms, field definitions, update schedule, and cost fit the project. Adding such an integration requires a specific provider choice and, where applicable, credentials supplied outside version control.
