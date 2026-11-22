# Tokopedia Sentiment Analysis — reproducible ML case study

This project evaluates three-class Indonesian sentiment classification under extreme class imbalance. It uses 65,543 Tokopedia reviews with dates and product IDs to compare classical text models, prevent common leakage, measure temporal and unseen-product performance, tune an explicit minority-recall policy, and quantify uncertainty.

The historical notebook remains as an exploration artifact only. Its 82.79% binary accuracy was measured after global downsampling and vocabulary fitting before splitting, so it is not comparable with the package benchmark.

## Why accuracy is insufficient

The source data is 97.6% positive. The selected temporal two-stage model reaches 81.68% accuracy after threshold tuning, but macro-F1 is only 44.91%. Negative recall is 71.43% at 20.24% precision; neutral recall is 69.84% at 7.24% precision. Detecting more rare reviews therefore creates many false alerts. The precision-constrained triage mode is the recommended interface: it routes only 440 of 8,374 test reviews to candidate queues instead of treating every threshold result as a final label.

## Evaluation design

1. Normalize Indonesian slang/noise while preserving negation.
2. Remove empty, duplicate, and contradictory normalized texts before splitting.
3. Use reviews through 2024-10-22 for training, 2024-10-23–2025-06-03 for validation, and 2025-06-04 onward for the untouched temporal test.
4. Compare MultinomialNB, ComplementNB, and class-balanced Logistic Regression with word/character features on validation macro-F1.
5. Compare single-stage and two-stage calibrated architectures.
6. Tune both high-recall classification thresholds and precision-constrained triage queues on validation only.
7. Report per-class metrics, PR AUC, macro-F1, calibration, 500-iteration bootstrap intervals, repeated product-group CV, product-disjoint holdout, 16 slices, three rolling backtests, and drift diagnostics.
8. Run an isolated IndoBERT-Lite experiment without making the large dependency/model part of normal CI.

The selected model is a two-stage balanced Logistic Regression over word and character TF-IDF. See [the full results](docs/RESULTS.md), [dataset card](docs/DATASET.md), and [model card](docs/MODEL_CARD.md).

## Setup

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are expected:

```bash
uv sync --locked --extra dev
```

`uv.lock` pins the complete dependency graph. CSVs, annotation batches, models, reports, credentials, and caches are ignored by Git.

## Run the complete benchmark

Place `reviewTokped2025.csv` locally, then run:

```bash
uv run tokopedia-sentiment benchmark \
  --data reviewTokped2025.csv \
  --artifact artifacts/case-study-model.joblib \
  --reports reports/case-study
```

Generated local outputs include `metrics.json`, confusion matrix, one-vs-rest precision-recall curves, reliability diagram, and a versioned model artifact containing provenance and threshold metadata.

Predict with the saved calibrated model and its validation-selected decision policy:

```bash
uv run tokopedia-sentiment predict \
  --artifact artifacts/case-study-model.joblib \
  "barang rusak dan saya sangat kecewa"
```

Only load trusted joblib artifacts; pickle-based loading can execute code.

For operational review routing, use triage instead of forced classification:

```bash
uv run tokopedia-sentiment triage \
  --artifact artifacts/case-study-model.joblib \
  "barang rusak dan saya sangat kecewa"
```

Generate a current-vs-history drift report:

```bash
uv run tokopedia-sentiment drift-report \
  --data reviewTokped2025.csv \
  --reference-end 2024-12-31 \
  --current-start 2025-01-01
```

## Build a human-labeled gold set

Proxy labels are derived exactly from rating, not human judgment. Create a balanced, ignored local batch:

```bash
uv run tokopedia-sentiment sample-annotation \
  --data reviewTokped2025.csv \
  --output annotations/review_batch.csv \
  --samples-per-rating 100
```

Follow [the annotation guide](docs/ANNOTATION_GUIDE.md). Two independent Indonesian-speaking annotators and adjudication are required; the project does not fabricate these labels.

After the pilot, create a second batch targeted at uncertainty, proxy disagreement, and decision-policy disagreement. The initial 500 review IDs are excluded automatically:

```bash
uv run tokopedia-sentiment sample-active-learning \
  --data reviewTokped2025.csv \
  --artifact artifacts/case-study-model.joblib \
  --output annotations/active_learning_batch.csv \
  --exclude-batch annotations/review_batch.csv
```

The blind and key files are separate. See [the human handoff](docs/HUMAN_HANDOFF.md) for the only remaining non-automatable work.

## Optional IndoBERT-Lite experiment

```bash
uv sync --locked --extra dev --extra bert
uv run tokopedia-sentiment transformer-benchmark \
  --data reviewTokped2025.csv \
  --output artifacts/indobert-lite \
  --report reports/transformer/metrics.json
```

The completed CPU smoke run underperformed the selected classical pipeline. The default two-epoch command should be rerun only with CUDA-enabled PyTorch or an adequate compute budget.

## Dataset-free demo and legacy baseline

```bash
uv run tokopedia-sentiment demo "gak enak bgt" "produk bagus"
uv run tokopedia-sentiment train --data reviewTokped.csv --calibrate
```

The demo uses 12 synthetic phrases and is behavior verification, not quality evidence. The `train` command preserves the earlier binary baseline for comparison.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests cover Indonesian preprocessing, temporal/group splits, threshold and triage behavior, two-stage probabilities, drift, active-learning sampling, transformer sampling, calibration, inference, reports, and artifact round-tripping. GitHub Actions installs only the locked `dev` extra and runs lint/format/test without downloading transformer weights.

## Repository map

```text
src/tokopedia_sentiment/   data, models, decision policy, evaluation, inference, CLI
scripts/                    convenience entry points
tests/                      unit, behavior, and synthetic integration tests
docs/DATASET.md             provenance, proxy labels, license, and data risks
docs/MODEL_CARD.md          intended use, performance, and operational limits
docs/RESULTS.md             temporal, group, bootstrap, and slice results
docs/ANNOTATION_GUIDE.md    human gold-set protocol
docs/HUMAN_HANDOFF.md       remaining annotator/license/remote-CI steps
.github/workflows/ci.yml    locked lint/format/test CI
```

## Remaining limitation

This is an offline research baseline, not a moderation system. The rare labels remain rating proxies, neutral precision is low, and no human gold set exists yet. The completed IndoBERT run is a CPU smoke experiment, not a full transformer comparison. Dataset use is restricted to its stated educational/research purpose unless the owner grants broader rights.

