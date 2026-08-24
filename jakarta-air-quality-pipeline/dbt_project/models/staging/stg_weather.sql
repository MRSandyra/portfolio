-- Cleaned, typed, deduplicated weather forecast slots.
with source as (
    select * from raw.weather_forecast
),

deduped as (
    select
        adm4_code,
        location_name,
        forecast_at,
        temperature_c,
        humidity_pct,
        wind_speed_ms,
        weather_desc,
        row_number() over (
            partition by adm4_code, forecast_at
            order by ingested_at desc
        ) as rn
    from source
    where forecast_at is not null
)

select
    adm4_code,
    location_name,
    forecast_at,
    temperature_c,
    humidity_pct,
    wind_speed_ms,
    weather_desc
from deduped
where rn = 1
