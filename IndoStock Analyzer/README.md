# IndoStock Analyzer

IndoStock Analyzer is an educational research tool that collects attributed market/news data and produces **exploratory indicators** for Indonesian equities. It does not produce Buy, Sell, Strong Buy, or Strong Sell recommendations.

> **Not financial advice.** This project is for educational and exploratory research only. It contains no bundled evidence that its indicators are suitable for investment decisions. Historical simulations are not an offer or recommendation to buy, sell, or hold a security and do not establish future performance. Verify source data independently and consult a qualified professional where appropriate.

## What changed in v2

The original single script is split into independently testable modules:

```text
indostock_analyzer/
├── scraper.py     # IndoPremier browser retrieval
├── data.py        # Yahoo Finance retrieval and technical calculations
├── sentiment.py   # Language-gated sentiment assessment
├── scoring.py     # Pure, bounded exploratory scoring functions
├── analyzer.py    # Pipeline orchestration and graceful failures
├── output.py      # JSON report writer
├── config.py      # Portable runtime configuration
└── cli.py         # Command-line interface
```

Every report includes source URLs, retrieval timestamps, source failures, component coverage, methodology status, and the disclaimer.

## Phase 1 reliability controls

- Successful public-source responses are cached as atomic JSON files for 300 seconds by default. Expired or malformed entries are discarded.
- Fundamental payloads must contain at least one scorable finite metric. Price history must be chronological, unique, finite, positive for closing prices, and non-negative for volume.
- Chrome performance logs are inspected for HTTP 429 responses and numeric `Retry-After` values. Page text detection remains as a fallback.
- Retry is reserved for potentially transient failures; invalid requests and invalid source payloads fail immediately.
- Runtime events are emitted as JSON logs to stderr. Reports retain structured `error_code` and `from_cache` fields.
- JSON reports are schema-checked and written through an atomic replace.

## Phase 2 universe and data quality controls

- The ticker universe is loaded from [`config/idx_tickers.json`](config/idx_tickers.json), with an `as_of` date, coverage level, source URL, retrieval timestamp, and per-symbol status.
- The bundled bootstrap snapshot contains 43 active symbols and declares `curated_subset`. Reports surface this warning instead of implying complete IDX coverage.
- Delisted symbols remain available for provenance but are excluded from headline extraction.
- Fundamental data records quote/financial currency, units, statement period, available metrics, missing metrics, extreme values, and staleness.
- Technical data records quote currency, adjustment policy, missing observations, calendar gaps, extreme daily returns, splits, dividends, and staleness.
- A stale fundamental or technical component is excluded from the composite indicator instead of being scored as current data.

Validate the bundled snapshot:

```bash
python -m indostock_analyzer.universe_cli validate config/idx_tickers.json
```

After downloading an official IDX CSV export, generate and validate a replacement snapshot:

```bash
python -m indostock_analyzer.universe_cli import-csv official_idx.csv config/idx_tickers.json \
  --as-of 2026-11-22 \
  --source-name "Indonesia Stock Exchange" \
  --source-url "https://www.idx.co.id/en/listed-companies/company-profiles/"
```

Accepted symbol headers include `Code`, `Kode`, `Stock Code`, `Kode Saham`, and `Symbol`. The importer rejects malformed or duplicate symbols and writes JSON atomically.

## Phase 3 Indonesian sentiment validation

- Sentiment is assessed per `(headline, ticker)`, so one headline is no longer assigned a single direction across every mentioned issuer.
- Local JSONL tooling validates independent annotations and reports label distribution, generalized agreement, accuracy, macro F1, class precision/recall/F1, confusion matrix, expected calibration error, and tagged failure slices.
- An Indonesian provider is loaded only as a separately installed `indostock.sentiment` plugin. Model artifacts and datasets stay outside this repository.
- Metric eligibility requires an attributed Indonesian financial-headline dataset, target-aware chronological holdout, immutable model identity, at least 300 evaluated records, macro F1 ≥ 0.75, every class recall ≥ 0.65, and ECE ≤ 0.15.
- Runtime activation additionally requires independent-review metadata and an operator-configured SHA-256 matching the exact report bytes. A generated report cannot activate its own provider.
- A failed or incomplete gate leaves Indonesian sentiment as `UNASSESSED`; it is never silently converted to neutral.

The annotation rules are in [`docs/sentiment-annotation-guide.md`](docs/sentiment-annotation-guide.md). Research scope, comparison protocol, activation thresholds, and cited dataset/model limitations are in [`docs/sentiment-model-evaluation.md`](docs/sentiment-model-evaluation.md).

## Sentiment compatibility audit

TextBlob's default `PatternAnalyzer` uses an English sentiment lexicon. That does not establish validity for Indonesian-language financial news. Because the current news source is predominantly Indonesian, the default `--news-language id` records sentiment as `UNASSESSED`; it does **not** silently turn unsupported text into a neutral score.

`--news-language en` enables TextBlob for explicitly English headlines, but those results remain uncalibrated and are labeled accordingly. A future Indonesian model should only replace this behavior after dataset provenance, task-specific validation, and calibration are documented.

Validate local annotations and generate a candidate report:

```bash
indostock-sentiment validate-annotations annotations/holdout.jsonl
indostock-sentiment evaluate \
  --annotations annotations/holdout.jsonl \
  --predictions evaluations/candidate-predictions.jsonl \
  --dataset-manifest config/sentiment_dataset_manifest.json \
  --model-manifest config/sentiment_model_manifest.json \
  --report-id candidate-YYYYMMDD \
  --output evaluations/candidate-report.json
```

The repository includes manifest templates only. It intentionally does not include or download a dataset, model weights, or generated predictions.

## Phase 4 point-in-time historical evaluation

- A separate backtest CLI evaluates frozen observations without changing runtime indicator labels.
- Look-ahead checks require every feature timestamp and universe snapshot to predate the signal.
- The manifest requires a point-in-time universe including inactive symbols, adjusted total returns, source/license attribution, an immutable file hash, a benchmark, temporal holdout with embargo, and the number of variants tried.
- Calibration is learned only from the pre-holdout period. Holdout results include Brier score, ECE, cross-sectional rank association, score buckets, benchmark-relative returns, turnover, costs, drawdown, information ratio, and a block-bootstrap interval.
- The research portfolio is equal-weighted and uses explicit commission, slippage, and tax assumptions. No current fee schedule is hard-coded.
- Every report remains `research_only_not_investment_validated`, even when sample-sufficiency checks pass.

Schema, methodology, and interpretation boundaries are documented in [`docs/backtesting-protocol.md`](docs/backtesting-protocol.md). Start from [`config/backtest_manifest.example.json`](config/backtest_manifest.example.json).

```bash
indostock-backtest \
  --observations backtests/observations.jsonl \
  --manifest config/backtest_manifest.json \
  --report-id indicator-v1-holdout-1 \
  --output backtests/report.json
```

## Phase 5 operations and artifact governance

- Analysis reports now carry a versioned artifact type, unique run ID, sanitized configuration fingerprint, software version, and SHA-256 hashes for local input artifacts.
- Analysis, sentiment-validation, and backtest writers produce atomic checksum sidecars.
- `indostock-audit` validates report contracts, verifies checksums, and checks the workspace for common secrets, generated data/model artifacts, symbolic links, and unexpectedly large files.
- CI covers Python 3.11 and 3.12, performs dependency and repository-policy checks, compiles/tests, builds a wheel, reinstalls it, and smoke-checks installed entry points.
- GitHub Actions dependencies are pinned to full commit SHAs and checkout credential persistence is disabled.
- No publish, deploy, signing, or release workflow was added; those require repository ownership and explicit release authority.

Operational boundaries and recovery guidance are documented in [`docs/operations-and-release.md`](docs/operations-and-release.md).

```bash
indostock-audit validate-report output/report.json
indostock-audit verify-checksum output/report.json
indostock-audit release-check .
```

Research-evidence recomputation uses the original frozen inputs:

```bash
indostock-audit verify-sentiment-evidence evaluations/report.json --annotations annotations/holdout.jsonl --predictions evaluations/predictions.jsonl --dataset-manifest config/sentiment_dataset_manifest.json --model-manifest config/sentiment_model_manifest.json
indostock-audit verify-backtest-evidence backtests/report.json --observations backtests/observations.jsonl --manifest config/backtest_manifest.json
```

## Phase 6 adversarial evidence audit

- Sentiment reports now distinguish `metrics_eligible` from runtime `active`. Locally generated evaluation reports are evidence candidates and remain inactive even when every metric threshold passes.
- Runtime sentiment configuration is fail-closed: provider name, report path, and the operator-approved report SHA-256 must all be supplied. A modified or substituted report is rejected before the plugin is loaded.
- Evaluation reports bind the annotation, prediction, dataset-manifest, and model-manifest hashes. Backtest reports bind the observation and manifest hashes.
- Offline artifact validation recomputes sentiment accuracy, class precision/recall/F1, and macro F1 from the confusion matrix. It also recomputes material portfolio summaries from the reported backtest periods and checks policy-flag consistency.
- A checksum passing means only that the current bytes match its sidecar. The artifact command reports internal consistency separately and explicitly does not claim independent verification of evidence origin or research conclusions.
- Regression tests cover forged high metrics, review claims without operator digest approval, checksum-valid semantic failures, future-input leakage, and tampered backtest summaries.

The evidence levels and review boundary are documented in [`docs/evidence-governance.md`](docs/evidence-governance.md).

## Phase 7 reproducible evidence verification

- Sentiment and backtest reports record the exact evaluation-policy snapshot used for their gates and sample checks.
- `indostock-audit verify-sentiment-evidence` verifies four bound input hashes, reruns the evaluation, and compares provenance, agreement, distribution, confusion-derived metrics, ECE, failure slices, and metric eligibility.
- `indostock-audit verify-backtest-evidence` verifies two bound input hashes, reruns temporal/leakage validation and the historical evaluator, and compares policy checks, calibration, association, portfolio statistics, individual periods, and limitations.
- A report whose checksum and internal shape are valid but whose ECE, association, calibration, or other research output was altered is rejected after recomputation.
- Successful output deliberately says `reproducibility`, not `independent validation`: the verifier uses the same codebase and cannot establish source-data truth or reviewer independence.

The commands and interpretation boundary are documented in [`docs/reproducible-verification.md`](docs/reproducible-verification.md).

## Phase 8 signed external-review approval

- A report can no longer activate Indonesian sentiment by declaring itself `independently_reviewed`; runtime approval requires a separate Ed25519-signed receipt.
- The receipt binds the exact report SHA-256, four evidence-input hashes, reviewer identity and organization, conflict disclosure, required review scope, methodology attestations, issue time, and expiry.
- Reviewer public keys live in an operator-managed trust store whose expected SHA-256 must also be configured, so silent trust-store substitution fails closed.
- Reviewer keys must be active, authorized for this receipt type, and valid for the entire receipt lifetime. Receipts expire after at most 366 days.
- The application prepares review requests and canonical unsigned payloads, but does not create reviewer keys or sign approvals.
- Runtime verifies report integrity, internal consistency, receipt bindings, trust policy, signature, and validity period before loading the sentiment plugin.

The reviewer/operator separation and receipt format are documented in [`docs/independent-review-protocol.md`](docs/independent-review-protocol.md).

## Phase 9 review quorum and revocation

- Runtime activation now requires a pinned bundle containing multiple signed receipts instead of one receipt.
- Quorum thresholds are stored in the pinned trust store and must require at least two distinct reviewers from at least two distinct organizations.
- Duplicate review IDs, reviewer IDs, or signing keys are rejected and cannot be counted twice.
- A separately pinned revocation registry can revoke review IDs, reviewer keys, or exact report hashes.
- The revocation registry is fail-closed when stale and must be refreshed at least every 31 days.
- Bundle, trust store, revocation registry, and validation report are independently SHA-256 pinned by runtime configuration.
- Verified quorum and its earliest receipt expiry are included in analyzer metadata without exposing file paths or public keys.

This improves approval governance but still cannot prove organizational independence or replace legal/data/model review.

## Phase 10 signed revocation chain and rollback checkpoint

- Phase 10 introduced schema v2 with one authorized Ed25519 revocation key; Phase 11 supersedes that runtime format with the multi-authority schema v3 below.
- Each registry carries a stable registry ID, monotonic sequence, and previous-registry SHA-256.
- A persistent local checkpoint records the highest accepted sequence and digest. Older sequences, same-sequence forks, skipped sequences, and broken hash chains fail closed.
- The checkpoint also records the last observation time and rejects a system-clock rollback of more than five minutes.
- Checkpoint updates use an atomic replacement and a cross-process lock. The first accepted signed registry is an explicitly pinned trust-on-first-use checkpoint.
- The application can create canonical signing bytes, but revocation private keys and signing remain external.

This reduces accidental or unprivileged rollback risk. It is not a public transparency log and cannot stop a privileged actor who can replace or delete both runtime configuration and local checkpoint state.

## Phase 11 revocation-authority quorum

- Runtime now accepts only revocation registry schema v3 with a `signatures` list; the legacy single-signature schema is rejected.
- The pinned trust policy must require at least two distinct revocation authorities from at least two distinct organizations.
- Every supplied signature must be valid, active, authorized, and within its key-validity period. Invalid extra signatures fail the complete registry.
- Duplicate authority IDs or signing keys cannot count twice, and organization identity must match the pinned trust store.
- Verified authority IDs and organizations are included in analyzer metadata without recording public keys or filesystem paths.

This removes one compromised revocation key as a sufficient approval path, while retaining the Phase 10 sequence, hash-chain, expiry, and checkpoint checks.

## Installation

Python 3.11 and Chrome/Chromium are required. Install the reproducible dependency set:

```bash
python -m pip install -r requirements.lock
```

For development and tests:

```bash
python -m pip install -r requirements-dev.lock
python -m pytest
```

The lock files are generated from `requirements.in` and `requirements-dev.in`. To update them deliberately:

```bash
uv pip compile --python-version 3.11 requirements.in --output-file requirements.lock
uv pip compile --python-version 3.11 requirements-dev.in --output-file requirements-dev.lock
```

## Usage

```bash
python stock_scraper.py
# or, after installing the package:
indostock-analyzer
```

By default Selenium Manager resolves a compatible ChromeDriver. There is no hard-coded driver path. If automatic resolution is unavailable, configure one explicitly:

```bash
python stock_scraper.py --driver-path /path/to/chromedriver
```

Useful options:

```text
--max-news 30
--timeout 20
--retries 2
--min-interval 2
--user-agent "OrganizationResearchBot/1.0 (contact: you@example.com)"
--news-language id
--output-dir output
--cache-dir .cache/indostock
--cache-ttl 300
--no-cache
--log-level INFO
--ticker-universe config/idx_tickers.json
--universe-max-age-days 45
--max-price-age-days 7
--max-fundamental-age-days 200
--extreme-return-percent 40
--sentiment-provider separately-installed-provider-name
--sentiment-validation-report evaluations/candidate-report.json
--sentiment-validation-sha256 64-lowercase-hex-digest-approved-by-operator
--sentiment-review-bundle review-receipts/candidate-bundle.json
--sentiment-review-bundle-sha256 64-lowercase-hex-bundle-digest
--sentiment-reviewer-trust-store config/sentiment_reviewer_trust_store.json
--sentiment-reviewer-trust-store-sha256 64-lowercase-hex-trust-store-digest
--sentiment-review-revocations config/sentiment_review_revocations.json
--sentiment-review-revocations-sha256 64-lowercase-hex-revocation-digest
--sentiment-review-checkpoint state/sentiment_review_revocations.checkpoint.json
--show-browser
```

Equivalent environment variables are `INDOSTOCK_CHROMEDRIVER`, `INDOSTOCK_MAX_NEWS`, `INDOSTOCK_TIMEOUT`, `INDOSTOCK_RETRIES`, `INDOSTOCK_BACKOFF`, `INDOSTOCK_MIN_INTERVAL`, `INDOSTOCK_USER_AGENT`, `INDOSTOCK_HEADLESS`, `INDOSTOCK_NEWS_LANGUAGE`, `INDOSTOCK_OUTPUT_DIR`, `INDOSTOCK_CACHE_ENABLED`, `INDOSTOCK_CACHE_DIR`, `INDOSTOCK_CACHE_TTL`, `INDOSTOCK_LOG_LEVEL`, `INDOSTOCK_TICKER_UNIVERSE`, `INDOSTOCK_UNIVERSE_MAX_AGE_DAYS`, `INDOSTOCK_MAX_PRICE_AGE_DAYS`, `INDOSTOCK_MAX_FUNDAMENTAL_AGE_DAYS`, `INDOSTOCK_EXTREME_RETURN_PERCENT`, `INDOSTOCK_SENTIMENT_PROVIDER`, `INDOSTOCK_SENTIMENT_VALIDATION_REPORT`, `INDOSTOCK_SENTIMENT_VALIDATION_SHA256`, `INDOSTOCK_SENTIMENT_REVIEW_BUNDLE`, `INDOSTOCK_SENTIMENT_REVIEW_BUNDLE_SHA256`, `INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE`, `INDOSTOCK_SENTIMENT_REVIEWER_TRUST_STORE_SHA256`, `INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS`, `INDOSTOCK_SENTIMENT_REVIEW_REVOCATIONS_SHA256`, and `INDOSTOCK_SENTIMENT_REVIEW_CHECKPOINT`.

## Retrieval policy

- Use a transparent user agent and add a real contact address for recurring use.
- Default pacing is one source interaction every two seconds.
- HTTP 429 and transient 5xx responses use bounded retry/backoff and respect `Retry-After` when exposed by the HTTP client.
- Cache files contain public source responses only and are excluded from version control.
- Browser and network operations have explicit timeouts.
- A source failure is recorded in the report and does not invent substitute data.
- Users remain responsible for reviewing source terms, robots policy, and permitted usage before running collection at scale.

## Indicator interpretation

The heuristic score stays within 1–5, but its labels are deliberately non-actionable:

| Score | Label |
|---:|---|
| ≥ 4.2 | upper exploratory range |
| ≥ 3.7 | positive exploratory range |
| ≥ 3.0 | mixed exploratory range |
| ≥ 2.3 | negative exploratory range |
| < 2.3 | lower exploratory range |

The nominal component weights are sentiment 20%, fundamental 50%, and technical 30%. Missing components are excluded rather than assigned a neutral value, and `coverage` states how much of the nominal weighting was actually available. These thresholds and weights are hypotheses only; they are not evidence of predictive performance.

## Known limitations

- No licensed point-in-time historical observation dataset or completed holdout report is bundled; the backtest framework therefore supplies no performance claim by itself.
- The bundled ticker universe is a 43-symbol bootstrap subset. Complete coverage requires importing a current official IDX export.
- Yahoo Finance fields can be delayed, incomplete, restated, or use provider-specific units. Debt/equity is normalized from Yahoo's percentage representation.
- Calendar-gap detection is a warning heuristic and does not include the full IDX holiday calendar.
- Corporate actions are recorded from fields supplied with Yahoo history; completeness still depends on the upstream source.
- Chrome performance logs improve HTTP status visibility, but browser logs can still be unavailable or incomplete in some environments.
- Scraper selectors depend on third-party markup and may require maintenance.
- Live source integration is not exercised by unit tests or CI.
- No permitted, independently annotated Indonesian financial-headline corpus or separately versioned model artifact is bundled. Indonesian sentiment therefore remains disabled by default.
- Historical evaluation still depends on the correctness and legal usability of user-supplied point-in-time data, benchmark series, corporate-action adjustments, and cost assumptions.
- The simple research simulator does not model liquidity capacity, partial fills, market impact, borrow availability, portfolio constraints, or taxes that vary by account/instrument.
- Failure-slice metrics are exposed for human review but do not yet have automatic per-slice activation thresholds.
- SHA-256 sidecars detect modification but are not signatures or proof of origin; CI does not currently generate signed provenance attestations.
- A valid Ed25519 receipt proves possession of a trusted private key and exact artifact binding; it cannot prove reviewer competence, organizational independence, or truthfulness of signed attestations.
- Trust-store selection and key rotation remain operator governance decisions. Compromise of a trusted reviewer private key requires trust-store replacement and receipt revocation outside this repository.
- Quorum counts distinct identifiers and organizations declared in signed receipts and the trust store; it cannot prove that the organizations are genuinely independent or free of common control.
- Revocation-authority quorum has the same organizational-identity boundary: it cannot detect collusion, common control, or coordinated key compromise across trusted authorities.
- Revocation is an operator-distributed, pinned snapshot rather than an online transparency log. A still-fresh but outdated snapshot may not contain a newly issued revocation.
- The local monotonic checkpoint is not a public transparency log. A privileged actor who can replace or delete both configuration and checkpoint state can re-bootstrap an older still-valid registry.
- The first checkpoint is trust-on-first-use around the separately pinned registry digest; new deployments do not independently discover the global latest sequence.
- Receipt expiry and revocation freshness still depend on an accurate system clock. The checkpoint detects backward movement after first use, but cannot establish externally trusted time.
- Internal consistency checks can catch contradictory summaries, but cannot establish that source observations, labels, benchmark data, or model predictions reflect reality.
- Reproducibility verification uses the same implementation that generated the report. Correlated implementation errors therefore require an independent implementation or external reviewer to detect.
- Reports generated before the evaluation-policy snapshot was added cannot be reproduced exactly by Phase 7 and must be regenerated from their frozen inputs.
- Live-source behavior, Chrome compatibility, and third-party markup remain outside offline CI coverage.

The source selection criteria and fallback boundary are documented in [docs/data-sources.md](docs/data-sources.md).

## Generated and sensitive files

`.gitignore` excludes `.env` files, credentials, browser binaries, caches, generated reports, datasets, and model artifacts. Review staged files before every commit; do not commit secrets or large generated assets.
