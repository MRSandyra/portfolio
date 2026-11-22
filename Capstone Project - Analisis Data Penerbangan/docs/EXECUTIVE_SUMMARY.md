# Executive Summary

## Business question

Which airlines, routes, and stop patterns are most represented in this flight-offer snapshot, and where can an analyst make a bounded pricing or assortment decision?

## Key insight

The legacy metric `jumlah tiket terjual` is not ticket sales: it counts non-null fare rows per airline. The notebook then sums that repeated count across every row, effectively squaring listing frequency. Stored output also reports 32 duplicates, missing route/date fields, nonnumeric fares such as `Alaska` and `$`, and an undocumented 2,461-row extract.

## Recommendation

Reframe the unit as a **flight offer listing**. Use airline and route counts only to prioritize coverage or competitive-fare review. Compare stop-pattern fares within the same route and collection window before any pricing action.

## Limitation

There is no booking, passenger, quantity, conversion, or revenue field, so demand and sales cannot be inferred. Source ownership, snapshot date, retrieval method, and license are unverified. Departure date, year, time zone, cabin, tax inclusion, and booking horizon are missing.

## Next decision

For pricing/assortment, obtain timestamped offers with full itinerary dates, cabin, taxes, and booking horizon. For sales/demand, obtain booking outcomes with passengers, revenue, exposure, and conversion. After the pipeline runs, `artifacts/EXECUTIVE_SUMMARY.md` supplies data-derived evidence, a recommendation, and a caveat for each insight.
