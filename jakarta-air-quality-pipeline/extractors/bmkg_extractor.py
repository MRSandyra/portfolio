"""
Extract weather forecast data for a Jakarta kelurahan from BMKG's public API
and load it into raw.weather_forecast.

No API key required. Rate limit: 60 requests/minute/IP. Docs + adm4 code
lookup: https://data.bmkg.go.id/prakiraan-cuaca/
"""
import os
import sys
import requests
import psycopg2

BMKG_BASE_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"


def get_db_connection():
    return psycopg2.connect(
        host=os.environ["WAREHOUSE_DB_HOST"],
        dbname=os.environ["WAREHOUSE_DB_NAME"],
        user=os.environ["WAREHOUSE_DB_USER"],
        password=os.environ["WAREHOUSE_DB_PASSWORD"],
    )


def fetch_forecast(adm4_code: str) -> dict:
    resp = requests.get(BMKG_BASE_URL, params={"adm4": adm4_code}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run():
    adm4_code = os.environ.get("BMKG_ADM4_CODE", "31.71.03.1001")
    data = fetch_forecast(adm4_code)

    location_name = data.get("lokasi", {}).get("desa", adm4_code)
    # BMKG nests forecast entries under data[0]["cuaca"] as a list of lists
    # (one inner list per day). Flatten them into individual 3-hourly rows.
    forecast_days = data.get("data", [{}])[0].get("cuaca", [])

    rows = []
    for day in forecast_days:
        for slot in day:
            rows.append((
                adm4_code,
                location_name,
                # "local_datetime" is WIB with no offset marker ("2026-08-19
                # 12:00:00"); inserting it into a TIMESTAMPTZ column makes
                # Postgres read it as UTC and silently shifts every forecast
                # +7h. "datetime" is the same instant, RFC3339 with a Z.
                slot.get("datetime"),
                slot.get("t"),      # temperature (Celsius)
                slot.get("hu"),     # humidity (%)
                slot.get("ws"),     # wind speed (m/s)
                slot.get("weather_desc"),
            ))

    if not rows:
        print("No forecast rows returned - check the adm4 code or BMKG response shape")
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO raw.weather_forecast
                    (adm4_code, location_name, forecast_at, temperature_c, humidity_pct, wind_speed_ms, weather_desc)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
        print(f"Loaded {len(rows)} weather forecast rows into raw.weather_forecast")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
