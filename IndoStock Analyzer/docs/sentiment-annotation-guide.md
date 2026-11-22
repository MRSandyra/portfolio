# Indonesian financial-headline annotation guide

This guide defines entity-aware sentiment labels for evaluating a model. It does not make an investment recommendation and does not establish that sentiment predicts returns.

## Unit of annotation

Annotate one `(headline, target ticker)` pair. A headline mentioning two issuers must produce two records because the same event can be positive for one issuer and negative for another. Judge the stated or strongly implied effect on the named issuer, not general market mood or the annotator's view of the stock.

Use only headlines whose collection, storage, and evaluation are permitted. Keep source URL, publication time, collection time, and license/terms evidence in the dataset inventory. Do not copy full articles when headline-level permission is all that exists.

## Labels

- `POSITIVE`: the headline states or strongly implies a favorable operational, financial, legal, or competitive effect for the target.
- `NEGATIVE`: the headline states or strongly implies an unfavorable effect for the target.
- `NEUTRAL`: factual information with no clear directional effect, or genuinely balanced effects.
- `IRRELEVANT`: the ticker was extracted but the headline is not about that issuer.
- `AMBIGUOUS`: context is insufficient or reasonable annotators cannot determine the direction. Do not force it to neutral.

Only the first three labels enter model metrics. `IRRELEVANT` and `AMBIGUOUS` remain in the audit trail and are excluded explicitly.

## Decision rules

1. Resolve the target entity before assigning direction. A supplier winning a contract may be positive for the supplier but not necessarily for the buyer.
2. Apply negation to the complete claim: “laba tidak tumbuh” is not positive merely because it contains “laba” and “tumbuh”.
3. Separate reported facts from forecasts, rumors, denials, and questions. Tag uncertain claims; do not treat a denied rumor as the underlying event.
4. Interpret numbers against their stated comparison. Revenue rising while profit falls can be mixed; use `NEUTRAL` or `AMBIGUOUS` unless the headline clearly foregrounds one effect.
5. Do not infer that “harga turun” is good or bad for the company without context (for example input price versus selling price).
6. Tag irony or figurative language and use `AMBIGUOUS` if the intended direction is not explicit.
7. Ignore investment-action language from a quoted analyst when it does not describe issuer impact; this tool is not being calibrated to reproduce recommendations.

Recommended tags include `negation`, `multiple_entities`, `finance_jargon`, `irony`, `numeric_context`, `forecast`, `rumor`, and `corporate_action`.

## Annotation process

- Use at least two independent annotators per record. Annotators must not see model predictions.
- Require an adjudicated label and retain each original annotation. A third trained adjudicator resolves disagreement.
- Pilot the guide, review disagreement examples, freeze the guide version, and then annotate the held-out set.
- Record annotator IDs as pseudonyms; never store credentials or unnecessary personal data.
- Deduplicate near-identical syndications before splitting. Use a chronological cutoff so later headlines form the test set; do not randomly leak rewrites of the same story across splits.
- Freeze dataset and model versions before evaluation. Report label balance, generalized inter-annotator agreement, confusion matrix, per-class recall/F1, calibration error, and tagged failure slices.

## JSONL schema

Each line must contain an object like:

```json
{
  "id": "immutable-record-id",
  "text": "Headline text",
  "target": "BBRI",
  "annotations": [
    {"annotator_id": "ann-01", "label": "POSITIVE"},
    {"annotator_id": "ann-02", "label": "NEUTRAL"}
  ],
  "adjudicated_label": "POSITIVE",
  "tags": ["numeric_context"]
}
```

Annotation and prediction JSONL files are ignored by Git. Validate locally with:

```bash
indostock-sentiment validate-annotations path/to/annotations.jsonl
```
