# Methodology

## Indicator and unit of analysis

The source is the World Bank **Poverty and Inequality Platform (PIP): Percentiles** database, 2021 purchasing-power-parity (PPP) edition, version `20260922_2021`. A distribution is uniquely identified by `country_code`, survey `year`, `reporting_level`, and `welfare_type`.

PIP reports 100 ordered groups. `avg_welfare` is average daily welfare per person in 2021 PPP dollars; `quantile` is the percentile's upper threshold; `pop_share` and `pop` are the percentile's population share and population; and `welfare_share` is the percentile's share of total welfare.

“Welfare” must not automatically be read as cash income. `welfare_type` is either **income** or **consumption**, depending on the underlying household survey. The generated report prints the type and does not treat mixed types as strictly comparable. PIP itself sometimes uses “income” as shorthand for communication, but this project uses “welfare” when discussing mixed series.

## Selection rules

The analysis selects `reporting_level == "national"`. Urban and rural rows are not pooled with national distributions. The committed sample uses the exact country-year pairs in `global_inequality.data.SAMPLE_SELECTION`; they were chosen to provide three observations per comparison country plus Indonesia 2019 and 2025. They are examples, not a representative sample of countries or years.

The report's “latest” table means the latest observation **in the committed sample**. It is not a globally harmonized reference year. Full-data users should choose a documented strategy suited to their question—for example, a common survey-year window—and report year gaps rather than silently comparing each country's latest survey.

## Missing data and validation

No imputation is performed. `quantile` is retained as source metadata but may be missing (in particular when an upper threshold is not reported); it is not an input to the Lorenz, Gini, or simulation calculations. The preparation step:

1. filters the requested reporting level and optional country/year selection;
2. converts required numeric fields, treating failed conversion as missing;
3. fails if any required analytical value is missing (`percentile`, `avg_welfare`, `pop_share`, `welfare_share`, or `pop`);
4. fails on duplicate percentile keys, non-positive population shares, negative welfare shares, or negative population/average welfare; and
5. requires exactly one row for each percentile 1–100 by default, verifies that population and welfare shares each sum to one within numerical tolerance, and verifies that average welfare is non-decreasing by percentile.

It never drops an individual incomplete bin, because that changes the represented distribution. The pinned PIP source has non-negative welfare shares; distributions with negative welfare are outside the classical Lorenz scope implemented here and are rejected explicitly.

## Lorenz curve and Gini coefficient

For bins ordered from lowest to highest welfare, let normalized population shares be \(p_i\) and welfare shares be \(w_i\). The Lorenz points are:

\[
P_k = \sum_{{i=1}}^k p_i, \qquad L_k = \sum_{{i=1}}^k w_i
\]

with \((P_0, L_0)=(0,0)\). The grouped-data Gini is computed by trapezoidal integration:

\[
G = 1 - \sum_{{k=1}}^n (L_k + L_{{k-1}})(P_k-P_{{k-1}}).
\]

Inputs are normalized by their positive totals. This absorbs harmless floating-point drift, while completeness checks prevent normalization from hiding missing bins. Because only grouped percentile points are available, within-bin inequality is not represented; results can differ slightly from a microdata or official headline Gini.

## Illustrative simulations

Both scenarios are **illustrative static simulations, not policy forecasts**:

- **Bottom-50 subsidy:** adds 0.5 2021 PPP dollars per person per day to percentile groups 1–50. Funding is assumed to come from outside the modeled distribution, so this scenario is not budget-neutral.
- **Top-10 tax / bottom-40 transfer:** mechanically reduces average welfare in groups 91–100 by 10% and transfers the arithmetic revenue equally per person to groups 1–40. Within this accounting exercise, total welfare is preserved.

The simulations do not model administrative costs, eligibility and take-up, behavioral responses, general-equilibrium effects, tax avoidance/evasion, incidence outside the measured welfare aggregate, uncertainty, or the institutional details of a real policy. A lower simulated Gini demonstrates only the arithmetic consequence of the stated assumptions.

## Attribution and license

Data source: World Bank (2026), *Poverty and Inequality Platform: Percentiles*, version `20260922_2021`, accessed 20 November 2026, [dataset catalog](https://datacatalog.worldbank.org/search/dataset/0063646/poverty-and-inequality-platform-pip-percentiles). The catalog labels the data CC BY 4.0. The source CSV checksum is recorded in `data/sample/README.md` and enforced by the downloader.

PIP notes that underlying distributions can be based on microdata, binned data, imputed/synthetic data, or grouped data fitted with a parametric Lorenz distribution. Those provenance differences are an additional limitation for cross-country comparisons.

