# Committed sample

`pip_percentiles_sample.csv` is a small, deterministic extract used for offline reproduction and CI. It contains all 100 percentile rows for these national distributions:

| Country | Years | Welfare type |
|---|---|---|
| Brazil (`BRA`) | 2001, 2011, 2024 | income |
| China (`CHN`) | 1999, 2010, 2022 | consumption |
| Indonesia (`IDN`) | 2000, 2010, 2019, 2025 | consumption |
| United States (`USA`) | 2000, 2010, 2024 | income |
| South Africa (`ZAF`) | 2000, 2010, 2022 | consumption |

Source metadata:

- World Bank PIP Percentiles, 2021 PPP
- PIP version: `20260922_2021`
- Catalog resource: `DR0090357/world_100bin.csv`
- Source size: 25,998,833 bytes
- Source SHA-256: `6d35e14b8cbbb3effe53d52127d874d593eb0960810841368a9f6e83ce3943d1`
- Accessed: 20 November 2026
- Catalog license: CC BY 4.0

To rebuild the sample from the full source:

```bash
uv run inequality-download
uv run inequality-sample
```

`data/raw/` is ignored so the 25 MB source is never committed. `inequality-download` verifies the exact checksum before accepting the file, and `inequality-sample` fails if any listed distribution is absent or incomplete.

