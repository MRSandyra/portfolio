# Jakarta Air Quality Pipeline

An hourly ELT pipeline that pulls air quality readings from the [OpenAQ v3 API](https://docs.openaq.org/api) and weather forecasts from [BMKG](https://data.bmkg.go.id/prakiraan-cuaca/) for Jakarta. It lands the raw data in Postgres, transforms it with dbt, and serves it through a Metabase dashboard. Everything runs locally with Docker Compose. No cloud account is needed, only a free OpenAQ API key.

## The problem

Jakarta is one of the most polluted large cities in the world, but public data about it is scattered. Pollutant readings live in OpenAQ. Weather lives in BMKG. Neither is easy to query next to the other, and there is no single place that keeps a history of either one.

This project builds that missing piece: a pipeline that extracts both sources on a schedule, models them into a small warehouse, and blocks every run behind data quality tests so bad data does not quietly reach the dashboard.

The most honest finding from this project is not a chart, it is a number: of the 27 OpenAQ stations within 25 km of central Jakarta, only 2 have reported real data in the last few days. Most of the city's "coverage" is sensors that stopped transmitting long ago but still return their last reading through the API. Getting from 27 down to a trustworthy 2 was most of the actual work. See [Data Quality Issues Found](#data-quality-issues-found) below.

## Architecture

```
                     +------------------+      +------------------+
                     |   OpenAQ v3 API  |      |     BMKG API      |
                     |  /locations      |      | /prakiraan-cuaca  |
                     |  /locations/{}/  |      |  (no auth needed, |
                     |        latest    |      |   60 req/min)     |
                     +--------+---------+      +---------+---------+
                              |                          |
   +----------------------------------------------------------------------+
   |  Airflow (LocalExecutor, runs hourly)                                |
   |                          v                          v                |
   |              +-------------------+      +-------------------+        |
   |              |  extract_openaq   |      |   extract_bmkg    |        |
   |              +---------+---------+      +---------+---------+        |
   |                        +-----------+--------------+                  |
   |                                    v                                 |
   |                            +---------------+                         |
   |                            |    dbt_deps   |  installs dbt-utils     |
   |                            +-------+-------+                         |
   |                                    v                                 |
   |                            +---------------+                         |
   |                            |    dbt_run    |                         |
   |                            +-------+-------+                         |
   |                                    v                                 |
   |                            +---------------+                         |
   |                            |   dbt_test    |  11 tests, blocks run   |
   |                            +-------+-------+                         |
   +------------------------------------|----------------------------------+
                                         v
   +----------------------------------------------------------------------+
   |  Postgres "warehouse"                                                |
   |                                                                      |
   |   raw.air_quality_readings ---+                                      |
   |   raw.weather_forecast --------+                                     |
   |                                v                                     |
   |                    staging.stg_air_quality   (view, deduplicated)    |
   |                    staging.stg_weather       (view, deduplicated)    |
   |                                v                                     |
   |                    marts.mart_aqi_hourly          (table)            |
   |                    marts.mart_aqi_daily_summary   (table)            |
   +----------------------------------|-------------------------------------+
                                       v
                               +---------------+
                               |   Metabase    |  localhost:3000
                               +---------------+
```

The layers follow the usual bronze, silver, gold pattern. `raw` is append only and is never edited by hand. `staging` types and deduplicates the data. `marts` is what the dashboard actually reads.

## Running it

Requires Docker and a free OpenAQ API key. Register at [explore.openaq.org](https://explore.openaq.org). BMKG needs no key.

```bash
cp .env.example .env
```

Paste your OpenAQ key into `.env`, then bring the stack up:

```bash
docker compose up -d
```

Give Airflow a minute or two to set up its metadata database, then use these:

| Service | URL | Login |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| Metabase | http://localhost:3000 | set up on first visit, see below |
| Postgres | `localhost:5432` | `airflow` / `airflow`, database `warehouse` |

The DAG runs on its own every hour. To force a run right away:

```bash
docker compose exec airflow-scheduler airflow dags trigger jakarta_air_quality_pipeline
```

To look at the output directly:

```bash
docker compose exec postgres psql -U airflow -d warehouse -c "select * from marts.mart_aqi_hourly;"
```

To run the dbt models and tests by hand:

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/dbt_project && dbt build --profiles-dir ."
```

For a full walkthrough of setting up the Metabase dashboard (adding the warehouse as a data source, building the charts, screenshotting for a portfolio), see `METABASE_WALKTHROUGH.md`.

For a one-time script that seeds the dashboard with 14 days of real historical readings instead of waiting for the hourly DAG to build up history on its own, see `BACKFILL_WALKTHROUGH.md` and `extractors/backfill_openaq_history.py`.

## Data Quality Issues Found

This is the part of the project worth talking about in an interview. Writing the pipeline was straightforward. Making it correct took much longer, and every issue below was found by looking closely at real API responses and real data, not by guessing.

### 1. dbt quietly renamed the schemas

The first `dbt run` said it succeeded, but the tables were nowhere to be found. `dbt_project.yml` said `+schema: marts`, the database already had a `marts` schema, and yet the models landed in `staging_marts` instead.

dbt's default `generate_schema_name` macro joins the profile's target schema with the custom schema instead of replacing it. The target was `staging`, the custom schema was `marts`, so the result was `staging_marts`, a schema dbt created on its own. Nothing failed. The tables existed, the run was green, and they were simply in the wrong place.

This is fixed by overriding the macro in `dbt_project/macros/generate_schema_name.sql` so the custom schema is used as written. This override is close to mandatory for any project that wants plain, literal schema names.

### 2. dbt needs its packages installed before every run

`packages.yml` lists `dbt-utils` as a dependency, but nothing installs it automatically. Without a `dbt deps` step running first, `dbt run` and `dbt test` both fail at the parsing stage with a "package not installed" error, since dbt tries to load every test definition, including ones that reference `dbt_utils`, before running anything.

Fixed by adding a `dbt_deps` task to the Airflow DAG that always runs before `dbt_run`.

### 3. `dict.get(key, {})` does not protect against `null`

The extractor crashed with `AttributeError: 'NoneType' object has no attribute 'get'` inside code that looked defensive:

```python
last_utc = location.get("datetimeLast", {}).get("utc")
```

The default value in `dict.get(key, default)` in Python only kicks in when the key is missing. Several OpenAQ locations return `"datetimeLast": null`. The key is present, its value is explicitly `None`, and the default never fires.

The pattern that actually works is `(d.get(key) or {}).get(...)`, since it falls back on any falsy value, not only a missing key. This is applied everywhere the extractor reads a nested field that might be null.

### 4. The `latest` endpoint does not say what it measured

OpenAQ v3's `/locations/{id}/latest` endpoint returns readings shaped like this:

```json
{"sensorsId": 17000278, "value": 4053.0, "datetime": {"utc": "2026-08-19T04:00:00Z"}}
```

There is no `parameter` field and no `unit` field. Just a number. An early version of the extractor defaulted the missing parameter to `pm25`, which meant a raw particle count reading of 4053 was being stored as 4053 micrograms per cubic meter of PM2.5, about seventy times the hazardous threshold.

The parameter and unit only live on the location object, under `sensors`, keyed by the same `sensorsId`. The extractor now builds a `sensorsId -> (parameter, unit)` map from the location and joins each reading to it, dropping any reading whose sensor it cannot identify. When an API gives back a bare number with no unit, the only safe move is to refuse to guess. A wrong default is worse than a missing row.

### 5. Dead stations keep answering

Even with the sensor mapping fixed, the data still looked wrong. Readings dated 2024 sat in a table that had just been filled, next to values like `-999`.

Two separate problems were mixed together here. `-999` is a common sentinel for "no data" from a sensor that is powered on but not reporting anything real, and no amount of averaging turns it into a real concentration. And the `latest` endpoint really means "the most recent value we ever received," not "a recent value." A monitor that died in April 2024 still returns its April 2024 reading forever, with nothing marking it as stale.

Sentinels are easy to catch: reject anything negative. Staleness needed two separate checks. Filtering only on the station's overall `datetimeLast` was not enough, because one station can have a live sensor and several dead ones at the same time. One station's location level timestamp looked current while four of its five sensors had not reported since April 2024. So staleness is now checked twice, once per location and again per individual reading, against that reading's own timestamp.

That single fix explains most of the gap between "27 stations near Jakarta" and the 2 that are actually saying something real.

### 6. One test, seven pollutants, very different scales

A range test was added on the `value` column to catch future sentinel values: accepted range 0 to 2000, sized for PM2.5 in micrograms per cubic meter.

That range is reasonable for PM2.5 and meaningless for everything else. Carbon monoxide is usually reported in a completely different scale, so a broken CO sensor could pass a PM2.5 shaped test without complaint, while a normal CO reading might not even make sense compared to a 2000 microgram ceiling. One test across a long table with a `parameter` column is really several different tests wearing one disguise.

Scoping it with `config: {where: "parameter = 'pm25'"}` makes the test actually mean something. In a long or tall schema, any range test on a value column needs to be scoped by whatever column defines its unit.

### 7. Widening the search radius did not work, and the reason was worth knowing

Two live stations felt thin, so the search radius was widened from 25,000 to 60,000 meters to reach the rest of the Jabodetabek area. The extract task started failing with an HTTP 422 error.

The response explained why: OpenAQ v3 hard caps the `radius` parameter at 25,000 meters. Before reverting, the underlying idea was checked another way, using the API's `bbox` parameter instead of `radius`. A bounding box across the whole Jabodetabek area returned only one more station than the 25 km radius search, and that one extra station had not reported in 169 days, so the staleness filter would have dropped it anyway.

The radius was set back to 25,000. The interesting part is not the cap itself. It is that "search a wider area" was the obvious fix, and the data showed that the search radius was never the actual bottleneck. Jakarta does not have more live public monitors further out. It has two.

### 8. A green pipeline with a join that matched nothing

This one was the hardest to catch, because nothing looked broken. Every task succeeded, every test passed, and `mart_aqi_hourly` had zero rows with weather data attached.

The cause was a timezone bug. BMKG returns each forecast time slot with three different timestamp fields:

```json
{"datetime": "2026-08-19T05:00:00Z",
 "local_datetime": "2026-08-19 05:00:00",
 "utc_datetime": "2026-08-19 05:00:00"}
```

The extractor was reading `local_datetime` first, which is in WIB time (UTC plus 7 hours) with no timezone marker attached to it at all. Once inserted into a `TIMESTAMPTZ` column, Postgres read that naive timestamp as if it were already UTC, so every forecast landed about 7 hours in the future with no error and no warning. Reading the `datetime` field instead, which carries an explicit `Z` for UTC, fixed it.

That exposed a second problem underneath. The mart's own comment said it joined each reading to the nearest same hour forecast, but the actual SQL was an equality join on `date_trunc('hour', ...)`. BMKG only publishes forecasts every 3 hours, so an hourly air quality reading only lines up with a BMKG slot about a third of the time even with perfect timestamps. The join is now a nearest match within plus or minus 3 hours instead of a strict equality match, which is what the comment claimed all along.

The real lesson here is about the tests. All ten tests passing told us nothing about this bug, because every one of them checked a column on a row that already existed. A row that never gets joined is not there to break a `not_null` or a range check, it is simply absent. So an eleventh test was added, `assert_weather_window_is_current`, that checks the forecast window starts near the current time instead of half a day away. It was confirmed to fail against the old, buggy data before being trusted.

### 9. The historical backfill endpoint hid its own timestamp

A separate one-time script (`extractors/backfill_openaq_history.py`) was added to seed the dashboard with 14 days of real historical readings, using OpenAQ's `GET /v3/sensors/{id}/measurements` endpoint, so the trend chart would not need the laptop running for days to fill in.

The first version of that script fetched real data (459 measurements came back) but inserted 0 rows, because every row was silently rejected. The bug was in the timestamp field, not the value. The code read `measurement.get("datetimeFrom")`, expecting it at the top level of each measurement, but the actual field lives nested inside `measurement["period"]["datetimeFrom"]["utc"]`. The top level lookup returned nothing every time, so every row was dropped as if it had no timestamp, even though the `value` field was populated correctly the whole time.

Once the path was corrected, all 459 historical readings loaded and the trend chart went from 2 dots to a real 14 day line. The lesson repeats the one from issue 4: do not assume a field's location from a summary of documentation. Print one real response and read it.

### 10. Two stations, two different low cost sensors, no shared calibration

The two live stations disagree sharply. `BMKG 1` reports PM2.5 in the 55 to 85 microgram range across the backfilled window, while `Jakarta` reports 6 to 17, a 4 to 5 times gap for two readings both labeled as the same city.

The first guess was that one station was a reference grade government monitor and the other was a cheap sensor. That guess was checked against the API and it was wrong. Querying each location's `owner`, `provider`, and `instruments` fields shows that `Jakarta` is a Clarity sensor node and `BMKG 1` is an AirGradient sensor node. Both are flagged `isMonitor: false` by OpenAQ itself, meaning neither one is a government reference monitor.

The real explanation is cross vendor disagreement. Low cost optical PM2.5 sensors from different manufacturers use different calibration curves and are known to disagree with each other and with reference instruments, especially in humid climates like Jakarta's. OpenAQ aggregates both vendors under one API without correcting for this.

Practical takeaway: this dashboard can show a reliable trend for one station over time, but comparing raw PM2.5 numbers between the two stations is not sound without independent calibration. Any chart that puts both stations side by side should say so.

## Current state

A representative run after the historical backfill:

| Metric | Value |
|---|---|
| OpenAQ stations found within 25 km | 27 |
| Stations dropped as stale (no report in 3+ days) | 25 |
| Live stations loaded | 2, `Jakarta` (id 1894639, Clarity sensor) and `BMKG 1` (id 6455006, AirGradient sensor) |
| Historical readings loaded by the one-time backfill | 459 (pm1 and pm25, last 14 days) |
| Live hourly readings loaded per run | 2 to 3 |
| BMKG forecast rows loaded per run | around 20 (3 hourly grid, 3 days ahead) |
| dbt tests | 11 passed, 0 failed |

Two working monitors is not a data volume problem waiting to be solved. It is the finding. A pipeline that reported more than that would be lying about how much public air quality coverage Jakarta actually has.

## Repo layout

```
dags/air_quality_pipeline_dag.py         Airflow DAG (extract x2, then deps, run, test)
extractors/openaq_extractor.py           OpenAQ v3 latest reading extractor
extractors/bmkg_extractor.py             BMKG weather forecast extractor
extractors/backfill_openaq_history.py    one time historical backfill, not part of the DAG
dbt_project/models/staging/              typed, deduplicated views
dbt_project/models/marts/                hourly and daily tables the dashboard reads
dbt_project/tests/                       custom singular tests
dbt_project/macros/generate_schema_name.sql   forces literal schema names
init-db.sql                              warehouse database, schemas, raw tables
docker-compose.yml                       Postgres, Airflow, Metabase
METABASE_WALKTHROUGH.md                  step by step dashboard setup guide
BACKFILL_WALKTHROUGH.md                  step by step guide for the historical backfill
```