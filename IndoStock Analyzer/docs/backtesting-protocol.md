# Point-in-time backtesting protocol

This evaluator is a historical research tool, not a trading system and not financial advice. It never changes the analyzer's exploratory labels or marks them as investment-validated.

## Why the controls are strict

Randomly splitting time series can train on future observations and evaluate on the past; chronological splits with a gap are therefore required. Backtest results are also vulnerable to trying many variants and reporting only the best one, and to selecting only securities that survived until today. These risks are discussed in the [scikit-learn time-series split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [Bailey et al. on backtest overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659), and [Brown et al. on survivorship bias](https://academic.oup.com/rfs/article-abstract/5/4/553/1590264).

The evaluator consequently requires:

- a point-in-time universe snapshot at every signal time;
- explicit confirmation that historical inactive/delisted securities are retained;
- every input's `available_at` timestamp at or before `signal_at`;
- corporate-action-adjusted total returns;
- a chronological calibration set, embargo gap, and untouched holdout;
- non-overlapping outcome/rebalance periods;
- an explicit total-return benchmark;
- explicit commission, slippage, and tax assumptions in basis points;
- disclosure of every strategy/configuration variant tried;
- an immutable observation-file SHA-256 and source/license provenance.

## Observation JSONL

Each line represents one ticker at one historical signal time:

```json
{
  "id": "immutable-id",
  "ticker": "BBRI",
  "signal_at": "2025-02-01T16:15:00+07:00",
  "outcome_start": "2025-02-03T09:00:00+07:00",
  "outcome_end": "2025-03-03T16:00:00+07:00",
  "inputs_available_at": {
    "news": "2025-02-01T15:00:00+07:00",
    "fundamental": "2025-01-28T00:00:00+07:00",
    "technical": "2025-02-01T16:00:00+07:00"
  },
  "universe_as_of": "2025-02-01T00:00:00+07:00",
  "universe_snapshot_id": "idx-2025-02-01",
  "eligible_at_signal": true,
  "score": 3.8,
  "forward_total_return": 0.025,
  "benchmark_total_return": 0.011
}
```

The analyzer itself does not create this historical dataset because current-source queries cannot reconstruct what was genuinely available at each past timestamp. Build it from licensed point-in-time sources and retain the raw-to-observation transformation audit trail outside this repository.

## Metrics

Calibration uses only the pre-holdout period and fixed score buckets. It estimates the empirical probability that a ticker's total return exceeds the configured benchmark. Holdout reporting includes Brier score and expected calibration error. Brier score combines calibration and discrimination effects, so it must not be interpreted alone; see the [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

The holdout also reports cross-sectional Spearman association, bucket returns, a top-fraction equal-weight research simulation, turnover, modeled cost, benchmark-relative returns, information ratio, drawdown, and a deterministic circular block-bootstrap interval. These are descriptive historical metrics, not a guarantee, recommendation, or statistical proof of future performance.

## Run

Start from `config/backtest_manifest.example.json`, calculate the SHA-256 of the frozen JSONL file, document non-zero cost assumptions where applicable, and run:

```bash
indostock-backtest \
  --observations backtests/observations.jsonl \
  --manifest config/backtest_manifest.json \
  --report-id indicator-v1-holdout-1 \
  --output backtests/report.json
```

The default evidence sufficiency checks require at least 100 calibration observations, 100 holdout observations, 12 holdout periods, three symbols per period, and 20 calibration observations in every score bucket. Passing those sample checks does not mean the indicator is useful or suitable for investment.
