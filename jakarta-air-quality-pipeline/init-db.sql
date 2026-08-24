-- Runs once, automatically, the first time the postgres container starts.
-- Creates a second database ("warehouse") separate from Airflow's own
-- metadata database ("airflow"), plus the three schemas for the
-- bronze/silver/gold (raw/staging/marts) layers.

CREATE DATABASE warehouse;

\connect warehouse

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;

CREATE TABLE IF NOT EXISTS raw.air_quality_readings (
    id              SERIAL PRIMARY KEY,
    location_id     TEXT,
    location_name   TEXT,
    parameter       TEXT,
    value           DOUBLE PRECISION,
    unit            TEXT,
    measured_at     TIMESTAMPTZ,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.weather_forecast (
    id              SERIAL PRIMARY KEY,
    adm4_code       TEXT,
    location_name   TEXT,
    forecast_at     TIMESTAMPTZ,
    temperature_c   DOUBLE PRECISION,
    humidity_pct    DOUBLE PRECISION,
    wind_speed_ms   DOUBLE PRECISION,
    weather_desc    TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
