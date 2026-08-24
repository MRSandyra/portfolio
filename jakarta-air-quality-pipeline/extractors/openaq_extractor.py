import os
import sys
from datetime import datetime, timedelta, timezone

import requests
import psycopg2

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
JAKARTA_LAT = -6.2088
JAKARTA_LON = 106.8456
# OpenAQ v3 hard-caps /locations?radius at 25 km - anything larger returns
# HTTP 422. Widening to cover all of Jabodetabek via bbox was tried and adds
# exactly one station (Bogor Selatan, last report 169 days ago), so it buys
# no live coverage. 25 km it is.
SEARCH_RADIUS_METERS = 25_000  
STALE_LOCATION_DAYS = 3        
STALE_READING_DAYS = 3         

RELEVANT_PARAMETERS = {"pm1", "pm25", "pm10", "co", "no2", "o3", "so2"}


def get_db_connection():
    return psycopg2.connect(
        host=os.environ["WAREHOUSE_DB_HOST"],
        dbname=os.environ["WAREHOUSE_DB_NAME"],
        user=os.environ["WAREHOUSE_DB_USER"],
        password=os.environ["WAREHOUSE_DB_PASSWORD"],
    )


def fetch_jakarta_locations(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{OPENAQ_BASE_URL}/locations",
        headers={"X-API-Key": api_key},
        params={
            "coordinates": f"{JAKARTA_LAT},{JAKARTA_LON}",
            "radius": SEARCH_RADIUS_METERS,
            "limit": 100,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_latest_for_location(api_key: str, location_id: int) -> list[dict]:
    resp = requests.get(
        f"{OPENAQ_BASE_URL}/locations/{location_id}/latest",
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def is_stale(location: dict) -> bool:
    last_utc = (location.get("datetimeLast") or {}).get("utc")
    if not last_utc:
        return True
    last_dt = datetime.fromisoformat(last_utc.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - last_dt) > timedelta(days=STALE_LOCATION_DAYS)


def is_valid_value(value) -> bool:
    return value is not None and value >= 0


def is_stale_reading(measured_at_utc: str | None) -> bool:
    if not measured_at_utc:
        return True
    measured_dt = datetime.fromisoformat(measured_at_utc.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - measured_dt) > timedelta(days=STALE_READING_DAYS)


def run():
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        print("OPENAQ_API_KEY is not set - see .env.example", file=sys.stderr)
        sys.exit(1)

    locations = fetch_jakarta_locations(api_key)
    print(f"Found {len(locations)} OpenAQ stations near Jakarta")

    rows = []
    skipped_stale_locations = 0
    skipped_invalid_readings = 0

    for loc in locations:
        if is_stale(loc):
            skipped_stale_locations += 1
            continue

        location_id = loc["id"]
        location_name = loc.get("name", "unknown")
        lat = (loc.get("coordinates") or {}).get("latitude")
        lon = (loc.get("coordinates") or {}).get("longitude")

        sensor_map = {
            sensor["id"]: (
                (sensor.get("parameter") or {}).get("name"),
                (sensor.get("parameter") or {}).get("units"),
            )
            for sensor in (loc.get("sensors") or [])
        }

        for reading in fetch_latest_for_location(api_key, location_id):
            value = reading.get("value")
            measured_at = (reading.get("datetime") or {}).get("utc")

            if not is_valid_value(value) or is_stale_reading(measured_at):
                skipped_invalid_readings += 1
                continue

            parameter, unit = sensor_map.get(reading.get("sensorsId"), (None, None))
            if parameter is None or parameter not in RELEVANT_PARAMETERS:
                skipped_invalid_readings += 1
                continue

            rows.append((
                str(location_id),
                location_name,
                parameter,
                value,
                unit,
                measured_at,
                lat,
                lon,
            ))

    print(f"Skipped {skipped_stale_locations} stale location(s) (no report in {STALE_LOCATION_DAYS}+ days)")
    print(f"Skipped {skipped_invalid_readings} invalid/unmapped reading(s)")

    if not rows:
        print("No readings returned - nothing to load")
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO raw.air_quality_readings
                    (location_id, location_name, parameter, value, unit, measured_at, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
        print(f"Loaded {len(rows)} air quality readings into raw.air_quality_readings")
    finally:
        conn.close()


if __name__ == "__main__":
    run()