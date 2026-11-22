# Reproducible three-class benchmark

Run date: **2026-11-20**  
Seed: `42`  
Dataset SHA-256: `49240e20754048882a0d9ff6e667cd00bf05007b3b63f2cc84fb665d07e14266`  
Selected pipeline: two-stage word/character TF-IDF + class-balanced Logistic Regression (`C=1.0`) with sigmoid calibration

## Selection and architecture

Six single-stage candidates were compared on temporal validation. Balanced Logistic Regression had the highest macro-F1 (`0.5432`). Using that feature/classifier family, a calibrated two-stage architecture was then compared fairly on the same validation set:

| Architecture | Macro-F1 | Balanced accuracy | Negative recall | Neutral recall | Positive recall |
|---|---:|---:|---:|---:|---:|
| Single-stage calibrated | 0.4271 | 0.3913 | 0.1481 | 0.0260 | 0.9996 |
| Two-stage calibrated | **0.4585** | **0.4132** | **0.1630** | **0.0779** | 0.9988 |

The two-stage model first separates positive from non-positive, then negative from neutral. All architecture and threshold choices were made before test evaluation.

## Classification policy on temporal test

The validation-selected rare-class thresholds were `0.01` for negative and neutral. The joint 70% validation recall target remained infeasible (`target_met=false`), although neutral validation recall reached 72.73%; negative validation recall was 52.59%.

Temporal test support: 140 negative, 126 neutral, and 8,108 positive reviews.

| Metric | Value | Bootstrap 95% interval |
|---|---:|---:|
| Accuracy | 0.8168 | 0.8081–0.8255 |
| Balanced accuracy | 0.7444 | 0.7113–0.7804 |
| Macro-F1 | 0.4491 | 0.4362–0.4631 |
| Macro average precision | 0.5782 | — |
| One-vs-rest macro ROC AUC | 0.9620 | — |
| Log loss | 0.0868 | — |
| Multiclass Brier score | 0.0437 | — |

| Class | Precision | Recall | Recall 95% interval | F1 |
|---|---:|---:|---:|---:|
| Negative | 0.2024 | 0.7143 | 0.6500–0.7895 | 0.3155 |
| Neutral | 0.0724 | 0.6984 | 0.6190–0.7778 | 0.1311 |
| Positive | 0.9982 | 0.8204 | 0.8118–0.8293 | 0.9006 |

Confusion matrix, rows actual and columns predicted (`negative`, `neutral`, `positive`):

```text
[[ 100,   37,    3],
 [  29,   88,    9],
 [ 365, 1091, 6652]]
```

This high-recall classification policy still creates too many false minority labels for autonomous use.

## Precision-constrained triage

The operational mode uses validation precision constraints, a maximum 10% queue fraction per rare class, a high-precision positive route, and explicit abstention when no route qualifies.

| Route | Rows | Test precision | Test recall/coverage |
|---|---:|---:|---:|
| Negative candidate | 351 | 0.3191 | recall 0.8000 |
| Neutral candidate | 89 | 0.2247 | recall 0.1587 |
| Positive | 7,934 | 0.9911 | recall 0.9698 |
| Abstain | 0 | — | fraction 0.0000 |

Only 440 of 8,374 reviews enter a human-review queue. This is the recommended interface: candidate queues are not final sentiment labels.

On the product-disjoint holdout, triage produced negative precision/recall `0.3851/0.5794`, neutral `0.1574/0.3426`, positive precision `0.9919`, and 18 abstentions.

## Product and temporal robustness

Product-disjoint classification achieved macro-F1 `0.4171`, balanced accuracy `0.7272`, negative recall `0.7009`, and neutral recall `0.6574`. Precision remained low (`0.1389` negative; `0.0640` neutral).

Two-stage **2×3 repeated product-group CV with default argmax** produced:

- macro-F1 `0.4582 ± 0.0123`;
- balanced accuracy `0.4184 ± 0.0086`;
- negative recall `0.2157 ± 0.0207`;
- neutral recall `0.0416 ± 0.0093`;
- positive recall `0.9979 ± 0.0005`.

This contrast shows why an explicit decision policy is needed and why its precision/recall trade-off must be reported separately from probability argmax performance.

Three rolling six-month backtests yielded macro-F1 `0.4195 ± 0.0269`. Unseen-product macro-F1 was `0.3366`, `0.4429`, and `0.3935` across the three windows. Rare-class support in each unseen-product slice was small, so these values are diagnostic rather than definitive.

## Drift

Comparing 2025 with history through 2024 produced no configured alerts. Category Jensen-Shannon divergence was `0.0245`, token-length PSI `0.0090`, and current OOV token rate `0.0297`. The later temporal test comparison also triggered no thresholds. This does not prove absence of semantic or label drift; the monitor covers observable prevalence, category, rating, length, and vocabulary changes.

## IndoBERT-Lite limited experiment

The optional runner uses `indobenchmark/indobert-lite-base-p1`, class-weighted loss, temporal validation/test, and training-only majority reduction. The available locked PyTorch wheel was CPU-only. A two-epoch run was stopped after 30 minutes without a completed epoch report. A reproducible one-epoch smoke run used all 1,065 minority training rows plus 1,065 positive rows:

| Metric | IndoBERT-Lite smoke | Selected classical pipeline |
|---|---:|---:|
| Validation argmax macro-F1 | 0.4024 | 0.4585 architecture comparison |
| Test macro-F1 after threshold policy | 0.2740 | **0.4491** |
| Test balanced accuracy | 0.4904 | **0.7444** |
| Test macro average precision | 0.4214 | **0.5782** |

The smoke model is not selected. A full CUDA-enabled fine-tune may behave differently and must be rerun before making a general transformer comparison.

## Active learning and human evidence

Two ignored blind batches now exist:

- 500 rating-stratified pilot reviews;
- 500 uncertainty/disagreement-selected reviews, excluding every pilot review ID.

Both have separate hidden key files. Until two humans independently annotate and adjudicate a frozen gold set, every benchmark above remains an evaluation against rating-derived proxy labels.
