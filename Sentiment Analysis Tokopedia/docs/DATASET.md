# Dataset card and provenance

## Primary dataset

- Name: **Tokopedia Product Reviews 2025**
- Publisher: Salman Abdurrahman (`salmanabdu`)
- Landing page: <https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025>
- Local filename: `reviewTokped2025.csv` (ignored; not redistributed)
- Accessed for this run: 2026-11-20
- File size: 24,012,348 bytes
- SHA-256: `49240e20754048882a0d9ff6e667cd00bf05007b3b63f2cc84fb665d07e14266`
- Raw shape: 65,543 rows × 13 columns
- Date range: 2015-11-18 through 2025-12-12
- Coverage: 5,521 products, 856 shops, and six product categories

The public Kaggle page did not expose readable license metadata to the automated audit. A secondary project data card identifies the license as **Educational & Research Purpose**: <https://github.com/hanny0w0/bigdata-tokopedia-review/blob/main/docs/milestones/tugas-1/DataCard.md>. Treat that as a restriction, verify the current owner terms directly, and do not redistribute the CSV or review excerpts without permission.

## Label definition

`sentiment_label` is exactly determined by rating in this file:

| Rating | Proxy label | Raw rows |
|---|---|---:|
| 1–2 | negative | 798 |
| 3 | neutral | 802 |
| 4–5 | positive | 63,943 |

It is not a human annotation of review text. Mixed sentiment, sarcasm, delivery complaints, and text-rating disagreement can therefore be mislabeled. The human annotation workflow deliberately hides rating and proxy label from annotators.

## Preparation audit

| Stage | Rows |
|---|---:|
| Raw | 65,543 |
| Empty after text normalization | 195 removed |
| Text appearing with contradictory proxy labels | 803 removed |
| Remaining normalized duplicates | 8,819 removed |
| Modeling dataset | 55,726 |

Final class distribution: 780 negative, 763 neutral, and 54,183 positive. No downsampling is applied to validation or test sets.

## Leakage controls

- Text normalization and contradictory-text removal occur before splitting.
- Vocabulary and model fitting happen inside training folds only.
- The primary test is chronological and later than all train/validation reviews.
- Repeated CV and the secondary holdout keep `product_id` disjoint.
- Training artifacts record source hash, split boundaries, class counts, seed, selected model, calibration status, and thresholds.

The temporal split still contains products also seen in training; this is intentional and disclosed. Product-disjoint performance is reported separately.

## Data and governance risks

- Extreme positive prevalence makes accuracy misleading and minority estimates noisy.
- Product URLs, names, IDs, and review text remain potentially sensitive marketplace data even without buyer names.
- Deduplication cannot detect paraphrases, coordinated reviews, spam, or copied templates.
- Historical reviews may not represent future language, products, or platform behavior.
- Category coverage is limited to six categories and is not population-representative.
- Review date is treated as event time, but collection methodology and sampling weights are not documented in the file.

## Legacy dataset

`reviewTokped.csv` is retained locally for the historical binary Food & Drink baseline. It is not used in the current three-class benchmark.

