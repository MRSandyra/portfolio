# Reproducible evidence verification

Phase 7 verifies that a research report can be regenerated from the exact files it claims to use. This is stricter than checking a report checksum or recalculating a few summary fields, but it is not independent validation and does not make any output financial advice.

## Sentiment report

Keep the frozen annotation file, prediction file, dataset manifest, and model manifest used by `indostock-sentiment evaluate`. Then run:

```bash
indostock-audit verify-sentiment-evidence evaluations/candidate-report.json \
  --annotations annotations/holdout.jsonl \
  --predictions evaluations/candidate-predictions.jsonl \
  --dataset-manifest config/sentiment_dataset_manifest.json \
  --model-manifest config/sentiment_model_manifest.json
```

The verifier checks all four SHA-256 bindings, validates the source records, verifies the annotation hash recorded by the dataset manifest, reconstructs the recorded policy, and reruns the evaluation. It compares dataset/model provenance, agreement, annotation summary, aggregate metrics, confusion matrix, ECE, tagged slices, and metric eligibility.

## Backtest report

Keep the frozen point-in-time observations and manifest used by `indostock-backtest`. Then run:

```bash
indostock-audit verify-backtest-evidence backtests/report.json \
  --observations backtests/observations.jsonl \
  --manifest config/backtest_manifest.json
```

The verifier checks both SHA-256 bindings, repeats manifest and leakage validation, reconstructs the recorded policy, and reruns the evaluator. It compares the dataset snapshot, policy checks, sample counts, economic calibration, association results, portfolio statistics, every reported period, and documented limitations.

## Failure interpretation

- `bound input SHA-256 mismatch` means a supplied input is not byte-for-byte the input named by the report.
- `recomputed section differs` means the supplied inputs are bound correctly but the report content does not match a fresh evaluation with the recorded policy and current implementation.
- `report lacks evaluation_policy` means an older report lacks enough information for exact recomputation; regenerate it from frozen inputs.
- A successful recomputation establishes only that this implementation deterministically reproduced the report from these inputs. It does not verify source truth, data licensing, reviewer identity, absence of shared implementation defects, or future investment performance.
