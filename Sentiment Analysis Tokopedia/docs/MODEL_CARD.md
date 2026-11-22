# Model card

## Model

Two-stage word unigram/bigram plus character 3–5-gram TF-IDF with class-balanced Logistic Regression (`C=1.0`). Stage one separates positive from non-positive; stage two separates negative from neutral. Both stages are sigmoid-calibrated with three-fold cross-validation. Classification thresholds and a separate precision-constrained triage policy are selected on validation data and stored with the model.

## Intended use

- Offline portfolio/research analysis of Indonesian Tokopedia review text.
- Ranking candidate negative or neutral reviews for human inspection.
- Comparing leakage-aware evaluation methods under extreme imbalance.

## Out of scope

- Automated moderation, seller penalties, customer profiling, or high-impact decisions.
- Claims about all Tokopedia reviews, other marketplaces, or future distributions.
- Treating probabilities as certainty or rating-derived labels as human truth.

## Inputs and outputs

Input is a non-empty Indonesian review string. Forced-classification output contains normalized text, threshold-policy class, probability of that chosen class, ordinary probability-argmax class, and every calibrated class probability. The recommended `triage` interface instead returns `negative_candidate`, `neutral_candidate`, `positive`, or `abstain`. Candidate routes require human review. Empty/noise-only inputs are rejected.

## Performance

See [RESULTS.md](RESULTS.md). Forced classification has high rare-class recall but low precision. Temporal triage improves negative precision to 31.91% and neutral precision to 22.47% while routing 440 reviews for manual review. It is still not suitable for autonomous decisions.

## Risks and mitigations

- **Proxy-label bias:** create an independently human-labeled gold set using [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).
- **Class imbalance:** report PR metrics and per-class recall/precision; avoid accuracy-only claims.
- **Product leakage:** report a separate product-disjoint holdout and repeated group CV.
- **Temporal drift:** keep the newest period as test and rerun evaluation for later data.
- **Unseen products:** report rolling seen/unseen-product slices in addition to a product-disjoint holdout.
- **Slang/typos:** include character n-grams and dedicated slices; continue expanding reviewed normalization cases.
- **Calibration drift:** monitor class prevalence, reliability diagrams, Brier score, and ECE before each deployment-like use.
- **Transformer overclaim:** the completed IndoBERT-Lite result is a one-epoch CPU smoke test and underperformed this model; rerun a full GPU experiment before drawing a general conclusion.

## Reproducibility

Seed `42`, exact dataset SHA-256, locked dependencies, model selection scores, thresholds, splits, and metrics are saved in the report/artifact. The large model and restricted dataset remain outside version control.

