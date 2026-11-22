# Human annotation guide

The source sentiment field is a rating-derived proxy, not a human judgment. Use `tokopedia-sentiment sample-annotation` to create an ignored blind batch plus a separate `_key.csv` containing rating/proxy labels. Have at least two Indonesian-speaking annotators label the blind file independently. Never give them the key during judgment.

## Labels

- **negative**: dissatisfaction, complaint, damage, failure, or clearly adverse experience dominates.
- **neutral**: factual/mixed content with no dominant positive or negative position, or balanced positive and negative clauses.
- **positive**: satisfaction, praise, recommendation, or clearly favorable experience dominates.
- **uncertain**: insufficient context, sarcasm, only an emoji/code, or annotator cannot decide. Keep `human_label` blank and set `uncertain=true`.

Negation changes polarity (`tidak enak` is negative). Delivery, packaging, product, and seller sentiment all count as review-level sentiment. Do not infer sentiment from star rating, product name, or category.

## Procedure

1. Give annotators only `review_batch.csv`; retain `review_batch_key.csv` for adjudication.
2. Randomize row order; the CLI already does this with seed 42.
3. Collect two independent labels per review.
4. Measure Cohen's kappa and raw agreement.
5. Adjudicate disagreements with a third reviewer or documented consensus.
6. Freeze the adjudicated set before model selection and never train on the final gold test portion.

Recommended starting batch: 100 examples per rating (500 total). Because rating is not the same as text sentiment, retain all disagreements between human and proxy labels for analysis rather than silently correcting them.

After the pilot guideline is stable, label `active_learning_batch.csv` using the same blind procedure. It contains 500 high-uncertainty/disagreement examples and excludes all pilot review IDs. Do not expose either `_key.csv` to annotators.

Agreement and gold-set commands:

```bash
uv run tokopedia-sentiment annotation-agreement \
  --annotator-a annotations/annotator_a.csv \
  --annotator-b annotations/annotator_b.csv

uv run tokopedia-sentiment build-gold-set \
  --annotator-a annotations/annotator_a.csv \
  --annotator-b annotations/annotator_b.csv \
  --adjudication annotations/disagreements.csv
```

The first command calculates raw agreement and Cohen's kappa and creates a disagreement queue. The second refuses invalid labels and combines agreements with completed adjudications. Keep a separate final gold-test portion frozen; do not use it for model or threshold selection.

Annotation batches contain review text and must remain under ignored `annotations/`. Review the dataset's current license before sharing them.

