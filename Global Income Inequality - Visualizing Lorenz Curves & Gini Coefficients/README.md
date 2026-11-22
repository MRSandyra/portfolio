# Global Income Inequality

A reproducible analysis of Lorenz curves and grouped-data Gini coefficients using the World Bank Poverty and Inequality Platform (PIP) Percentiles database. Computation, data preparation, static visualization, and illustrative redistribution scenarios live in tested Python modules; the notebook is a thin walkthrough of the same package.

> **Terminology:** PIP welfare can be income or consumption. This project calls it *welfare* unless a specific series is identified. The redistribution scenarios are illustrative arithmetic simulations, **not policy forecasts**.

## Reproduce the committed report

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked --group dev
uv run pytest -q
uv run inequality-report
```

Open `report/index.html`. The command uses the committed 1,600-row sample, so report reproduction is offline after dependencies are installed. Dependencies and transitive versions are fixed in `uv.lock`; Python 3.11–3.13 is supported and CI uses Python 3.12.

## Refresh from the pinned official source

```bash
uv run inequality-download
uv run inequality-sample
uv run inequality-report
```

The downloader fetches PIP Percentiles 2021 PPP, version `20260922_2021`, and rejects any SHA-256 mismatch. The 25 MB source is written under ignored `data/raw/`; only the documented deterministic sample is versioned. See [data/sample/README.md](data/sample/README.md).

## Project layout

```text
src/global_inequality/
  data.py          # download, checksum, selection, fail-fast validation
  lorenz.py        # Lorenz and grouped Gini computation
  simulation.py    # explicitly illustrative arithmetic scenarios
  visualize.py     # headless static figures
  report.py        # deterministic HTML/CSV report generation
tests/              # mathematical invariants, validation, accounting tests
data/sample/        # small attributed offline sample
docs/methodology.md # definitions, missing-data policy, selection, limitations
report/             # generated static report and assets
```

## Method summary

- Select national distributions only; urban/rural observations are not mixed in.
- Identify a distribution by country, survey year, reporting level, and welfare type.
- Require exactly 100 unique percentile rows and all analytical numeric values. Do not impute or silently drop incomplete bins; a missing unused `quantile` threshold is allowed.
- Normalize positive population and welfare shares, prepend `(0, 0)`, and integrate the Lorenz polygon with the trapezoid rule: `Gini = 1 - 2 × area`.
- Print welfare type and year. The report does not imply that mixed income/consumption definitions or non-harmonized survey years are directly comparable.

The full formula, indicator definition, selection rationale, source attribution, and limitations are in [docs/methodology.md](docs/methodology.md).

## Illustrative scenarios

For the Indonesia 2019 consumption distribution, the report demonstrates:

1. an externally funded 0.5 2021 PPP dollars/person/day addition to the bottom 50%; and
2. a static 10% reduction in top-decile welfare transferred arithmetically per person to the bottom 40%.

These exercises omit behavior, administration, eligibility, financing incidence, equilibrium effects, avoidance, and uncertainty. Their outputs must not be presented as predicted real-world policy impacts or evidence that one policy is superior.

## Data source and attribution

World Bank (2026), *Poverty and Inequality Platform: Percentiles*, version `20260922_2021`, accessed 20 November 2026. [Dataset catalog](https://datacatalog.worldbank.org/search/dataset/0063646/poverty-and-inequality-platform-pip-percentiles). Catalog license: CC BY 4.0.

PIP documents that percentile distributions may originate from microdata, binned data, imputed/synthetic data, or fitted grouped data. Grouped estimates also omit within-bin inequality; recomputed values can therefore differ from microdata or official headline Gini estimates.

## Notebook

`Global Income Inequality.ipynb` is an output-free walkthrough that imports the package and calls the same report builder. It contains no Colab path or duplicate analytical implementation. The command-line workflow remains the CI reproduction entry point.

