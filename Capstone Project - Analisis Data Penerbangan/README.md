# Flight Offer Analysis Capstone

A recruiter-facing data analysis project with reproducible cleaning, feature engineering, quality checks, business analysis, visualization, and reporting.

The source rows are **flight offer listings**, not ticket sales. Every generated insight is paired with a bounded recommendation and an explicit limitation.

## Business question

Which airlines, routes, and stop patterns are most represented in this snapshot, and where can a commercial analyst make a defensible pricing or assortment decision?

## Repository structure

```text
src/flight_analysis/
  schema.py          # source-column contract
  cleaning.py        # cleaning and rejected-row audit
  features.py        # fare, duration, stop, clock, and date parsers
  quality.py         # schema, duplicate, missing, fare, and date checks
  analysis.py        # KPI/airline/route/stop tables and narrative
  visualization.py   # decision-focused charts
  pipeline.py        # end-to-end orchestration
tests/               # unit tests and a small synthetic fixture
docs/                # dictionary, provenance/license, executive summary
.github/workflows/   # CI unit and CLI smoke tests
Capstone_Projek.ipynb # legacy notebook retained for audit history
```

## Reproduce from a clean environment

Prerequisites: Python 3.11 or 3.12 and a permitted local copy of `flight.csv`. The dataset is not committed because its source and license are unverified.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m flight_analysis run --input data/raw/flight.csv --output-dir artifacts --clean-output
```

On macOS/Linux, activate with `source .venv/bin/activate`. `--clean-output` removes only known pipeline-generated filenames before rebuilding them.

## Outputs

The pipeline writes cleaned and rejected-row CSVs, a JSON quality report, KPI/airline/route/stop tables, three charts, and a dataset-derived executive summary. Review the [data dictionary](docs/DATA_DICTIONARY.md), [source/license status](docs/DATA_SOURCE_AND_LICENSE.md), and [audit executive summary](docs/EXECUTIVE_SUMMARY.md) first.

## Quality policy

Checks cover required schema, exact duplicates, missing values, nonnumeric/non-positive fares, travel duration, stop labels, and yearless arrival dates. Chronological arrival-after-departure logic is explicitly marked **not testable** because no departure date, year, or time-zone context exists.

## Tests and CI

```powershell
python -m pytest
python -m flight_analysis run --input tests/fixtures/flight_sample.csv --output-dir artifacts/smoke --clean-output
```

CI installs the lock file on Python 3.11, runs unit tests, and executes the CLI against the synthetic fixture. It never requires or uploads private raw data.

## Legacy notebook audit

The refactor corrects the notebook's decision risks: listing frequency was labeled as ticket sales; the repeated group count was summed and therefore squared; a global missingness threshold replaced an explicit schema; derived values were backfilled across unrelated rows; invalid fares/date logic were not validated; and charts lacked bounded business recommendations.
