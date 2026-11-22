"""Business-oriented aggregations and defensible narrative generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisTables:
    kpis: pd.DataFrame
    airlines: pd.DataFrame
    routes: pd.DataFrame
    stops: pd.DataFrame


def build_analysis_tables(frame: pd.DataFrame) -> AnalysisTables:
    if frame.empty:
        raise ValueError("No valid rows remain after cleaning.")

    complete_routes = frame[frame["route"].notna()].copy()
    kpis = pd.DataFrame(
        [
            ("valid_offer_count", len(frame), "Valid unique offer rows; not ticket sales."),
            ("airline_count", frame["airline"].nunique(), "Distinct airline labels."),
            ("route_count", complete_routes["route"].nunique(), "Distinct complete airport pairs."),
            ("median_fare_usd", round(float(frame["fare_usd"].median()), 2), "Median listed fare."),
            (
                "complete_route_share",
                round(float(frame["route"].notna().mean()), 4),
                "Share of valid offers with both airports.",
            ),
            (
                "dated_offer_share",
                round(float(frame["arrival_month"].notna().mean()), 4),
                "Share of valid offers with a parseable arrival month/day.",
            ),
        ],
        columns=["metric", "value", "definition"],
    )

    airlines = (
        frame.groupby("airline", as_index=False)
        .agg(
            offer_count=("airline", "size"),
            median_fare_usd=("fare_usd", "median"),
            min_fare_usd=("fare_usd", "min"),
            max_fare_usd=("fare_usd", "max"),
            route_count=("route", "nunique"),
        )
        .sort_values(["offer_count", "airline"], ascending=[False, True])
        .reset_index(drop=True)
    )
    airlines["offer_share"] = (airlines["offer_count"] / len(frame)).round(4)

    routes = (
        complete_routes.groupby(
            ["route", "departure_airport", "destination_airport"], as_index=False
        )
        .agg(
            offer_count=("route", "size"),
            airline_count=("airline", "nunique"),
            median_fare_usd=("fare_usd", "median"),
            min_fare_usd=("fare_usd", "min"),
            max_fare_usd=("fare_usd", "max"),
            median_travel_minutes=("travel_minutes", "median"),
        )
        .sort_values(["offer_count", "route"], ascending=[False, True])
        .reset_index(drop=True)
    )

    stops = (
        frame.groupby("stop_count", as_index=False)
        .agg(
            offer_count=("stop_count", "size"),
            median_fare_usd=("fare_usd", "median"),
            fare_q1_usd=("fare_usd", lambda values: values.quantile(0.25)),
            fare_q3_usd=("fare_usd", lambda values: values.quantile(0.75)),
            median_travel_minutes=("travel_minutes", "median"),
        )
        .sort_values("stop_count")
        .reset_index(drop=True)
    )
    return AnalysisTables(kpis=kpis, airlines=airlines, routes=routes, stops=stops)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_executive_summary(
    clean: pd.DataFrame,
    rejected_count: int,
    tables: AnalysisTables,
) -> str:
    """Render insights with a recommendation and caveat beside each claim."""
    top_airline = tables.airlines.iloc[0]
    valid_count = len(clean)
    rejection_share = rejected_count / (valid_count + rejected_count)
    complete_route_share = float(clean["route"].notna().mean())
    dated_share = float(clean["arrival_month"].notna().mean())

    stop_lines = [
        f"{int(row.stop_count)} stop: ${row.median_fare_usd:,.0f} (n={int(row.offer_count):,})"
        for row in tables.stops.itertuples(index=False)
    ]
    stop_evidence = "; ".join(stop_lines)

    return f"""# Executive Summary

## Business question

Which airlines, routes, and stop patterns are most represented in this snapshot of flight offers, and where can a commercial analyst make a bounded pricing or assortment decision?

## Decision context

The unit of analysis is one scraped/listed flight offer—not a booking, passenger, or ticket sale. The pipeline retained **{valid_count:,}** valid unique offers and rejected **{rejected_count:,}** rows ({_percent(rejection_share)} of input after schema selection). Median listed fare is **${clean['fare_usd'].median():,.0f}**.

## Key insights and recommendations

1. **Availability concentration:** {top_airline.airline} represents {int(top_airline.offer_count):,} offers ({_percent(float(top_airline.offer_share))}) in this extract.
   - **Recommendation:** use this as a coverage signal when selecting airlines for a manual fare-benchmark review; do not allocate sales or marketing budget from this ranking.
   - **Limitation:** offer frequency can reflect scraper coverage, not customer demand.

2. **Fare and stops:** {stop_evidence}.
   - **Recommendation:** use these figures only to identify stop categories for a route-controlled follow-up comparison.
   - **Limitation:** the aggregate is confounded by route, airline, cabin, and booking horizon; it does not estimate the causal price impact of a stop.

3. **Decision-ready completeness:** {_percent(complete_route_share)} of retained offers have both route endpoints and {_percent(dated_share)} have a parseable arrival month/day.
   - **Recommendation:** restrict route-level decisions to complete routes and require a new extract with full dates before seasonal or time-series planning.
   - **Limitation:** the source omits departure date, year, time zone, cabin, booking timestamp, and a verified sales outcome.

## Overall limitation

Source ownership, retrieval method, snapshot date, and license are not documented in the legacy notebook. The data must not be redistributed or used for external operational decisions until provenance and permission are confirmed. Chronological arrival-after-departure validation is not possible with the available fields.

## Next decision

Decide whether this project will support **competitive offer monitoring** or **sales/demand analysis**. For the former, acquire a licensed, timestamped extract with complete itinerary context. For the latter, replace this listing dataset with booking/conversion data before making commercial recommendations.
"""

