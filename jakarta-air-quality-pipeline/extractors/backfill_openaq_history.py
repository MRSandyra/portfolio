import os
import sys
from datetime import datetime, timedelta, timezone

import requests
import psycopg2

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
BACKFILL_DAYS = 14

KNOWN_LIVE_LOCATION_IDS = [1894639, 6455006]

RELEVANT_PARAMETERS = {"pm1", "pm25", "pm10", "co", "no2", "o3", "so2"}


def get_db_connection():
    return psycopg2.connect(
        host=os.environ["WAREHOUSE_DB_HOST"],
        dbname=os.environ["WAREHOUSE_DB_NAME"],
        user=os.environ["WAREHOUSE_DB_USER"],
        password=os.environ["WAREHOUSE_DB_PASSWORD"],
    )


def fetch_location(api_key: str, location_id: int) -> dict:
    resp = requests.get(
        f"{OPENAQ_BASE_URL}/locations/{location_id}",
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0] if results else {}


def fetch_sensor_history(api_key: str, sensor_id: int, datetime_from: str, datetime_to: str) -> list[dict]:
    all_results = []
    page = 1
    while True:
        resp = requests.get(
            f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
            headers={"X-API-Key": api_key},
            params={
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 1000,
                "page": page,
            },
            timeout=30,
        )
        if resp.status_code == 422:
            print(f"OpenAQ rejected the request params: {resp.text}", file=sys.stderr)
        resp.raise_for_status()

        body = resp.json()
        results = body.get("results", [])
        all_results.extend(results)

        found = body.get("meta", {}).get("found", 0)
        if len(all_results) >= found or not results:
            break
        page += 1

    return all_results


def run():
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        print("OPENAQ_API_KEY is not set - see .env.example", file=sys.stderr)
        sys.exit(1)

    datetime_to = datetime.now(timezone.utc)
    datetime_from = datetime_to - timedelta(days=BACKFILL_DAYS)
    dt_from_str = datetime_from.isoformat().replace("+00:00", "Z")
    dt_to_str = datetime_to.isoformat().replace("+00:00", "Z")

    rows = []
    for location_id in KNOWN_LIVE_LOCATION_IDS:
        location = fetch_location(api_key, location_id)
        if not location:
            print(f"Could not fetch location {location_id}, skipping")
            continue

        location_name = location.get("name", "unknown")
        lat = (location.get("coordinates") or {}).get("latitude")
        lon = (location.get("coordinates") or {}).get("longitude")

        for sensor in (location.get("sensors") or []):
            parameter = (sensor.get("parameter") or {}).get("name")
            unit = (sensor.get("parameter") or {}).get("units")
            if parameter not in RELEVANT_PARAMETERS:
                continue

            sensor_id = sensor["id"]
            print(f"Fetching {BACKFILL_DAYS}d history for {location_name} / {parameter} (sensor {sensor_id})")

            history = fetch_sensor_history(api_key, sensor_id, dt_from_str, dt_to_str)
            print(f"  -> {len(history)} raw measurement(s) returned")

            for measurement in history:
                value = measurement.get("value")
                if value is None or value < 0:
                    continue
                measured_at = ((measurement.get("period") or {}).get("datetimeFrom") or {}).get("utc")
                if not measured_at:
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

    print(f"Total valid historical readings to insert: {len(rows)}")
    if not rows:
        print("Nothing to insert.")
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
        print(f"Inserted {len(rows)} backfilled rows into raw.air_quality_readings")
    finally:
        conn.close()


if __name__ == "__main__":
    run()