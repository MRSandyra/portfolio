# Operations and release boundary

This project is an exploratory research application. Operational hardening improves reproducibility and failure visibility; it does not make the indicators investment-validated or establish that the application is suitable for unattended trading.

## Artifact contract

Generated analysis, sentiment-validation, and backtest reports use `schema_version: 1` and an explicit `report_type`. The offline validator rejects unknown versions and prevents a backtest artifact from declaring a status other than `research_only_not_investment_validated`.

Analysis reports include:

- a unique run ID and UTC timestamps;
- application and Python versions;
- a sanitized configuration snapshot and SHA-256 fingerprint;
- input filenames, byte sizes, roles, and SHA-256 hashes;
- no environment variables, absolute paths, user-agent contents, or credentials.

Every report writer creates a `<report>.sha256` sidecar. Validate and verify without network access:

```bash
indostock-audit validate-report output/report.json
indostock-audit verify-checksum output/report.json
```

The checksum detects accidental or subsequent modification. It is not a digital signature and does not prove who produced the file. `validate-report` checks format and selected internal arithmetic consistency; it does not independently verify the evidence origin or research conclusion.

Sentiment and backtest reports bind their material input files by SHA-256. The sentiment runtime separately requires the operator to configure the exact approved report digest. These hashes prevent unnoticed substitution only when the expected values were obtained through a trustworthy review channel; they cannot make incorrect input data correct.

For research artifacts, also run the matching Phase 7 recomputation command with the original frozen inputs. This verifies more than the checksum and report-shape checks, but remains same-code reproducibility rather than an independent implementation or audit.

Phase 8 review receipts use detached Ed25519 signatures and an operator-pinned public-key trust store. Private reviewer keys are external credentials: do not generate, store, or commit them in this repository. A valid signature proves possession of the trusted key, not reviewer competence or truth of the review conclusion.

Phase 9 additionally pins a multi-receipt quorum bundle and a time-bounded revocation registry. Operational checks must refresh the revocation snapshot before `next_update_at`; allowing it to expire intentionally disables Indonesian sentiment until an operator supplies and pins a newer snapshot.

Phase 10 requires each revocation registry to be signed by an authorized external key and linked to its predecessor. Preserve the configured checkpoint as durable state, back it up with access controls, and apply every registry sequence in order. Do not delete or replace a checkpoint merely to bypass rollback, fork, chain-gap, or clock-rollback failures. A stale lock file after a confirmed process crash requires operator review before removal.

Phase 11 replaces the single revocation signature with a quorum from independently administered authorities. Keep their private keys in separate external key-management boundaries, review organization assignments in the pinned trust store, and do not collect multiple authority keys in one runtime or signing host. Schema v2 registries must be regenerated and co-signed as schema v3 before upgrading runtime activation.

## Release check

Run before staging or packaging:

```bash
indostock-audit release-check .
```

The check reports common credential assignments, `.env`/key files, datasets, models, JSONL/CSV/Parquet files, caches, generated output, symbolic links, and files larger than 5 MB. It reports filenames only for possible secrets. This is a narrow policy check, not a substitute for repository-host secret scanning, code review, dependency review, or a dedicated security assessment.

## Continuous integration

CI runs on Python 3.11 and 3.12, installs the lock file, checks dependency consistency, runs the release policy before code execution, compiles source, runs unit tests, builds a wheel, reinstalls it, and smoke-checks installed entry points.

GitHub-maintained actions are pinned to full commit SHAs and checkout persistence is disabled. GitHub recommends full-SHA pinning when action workflow lockfiles are not in use because tags can move; see the [GitHub secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

CI does not publish packages, create releases, deploy services, access credentials, or attest artifacts. Those steps require a real Git repository, owner-approved release policy, protected environments, and explicit authority. GitHub documents artifact attestations as signed provenance linked to workflow and repository identity; see [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations).

## Incident and failure handling

- Preserve the JSON report, checksum, structured stderr logs, input hashes, and exact package/lock versions.
- Do not retry invalid payloads indefinitely or silently replace failed sources with differently defined providers.
- Treat upstream markup/schema changes, incomplete browser logs, stale data, checksum failures, or provenance mismatches as unavailable evidence.
- Rotate any credential outside this project if it is accidentally exposed; deleting it from a working tree is not sufficient revocation.
