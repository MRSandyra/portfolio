-- Cleaned, typed, deduplicated air quality readings.
with source as (
    select * from raw.air_quality_readings
),

deduped as (
    select
        location_id,
        location_name,
        lower(parameter) as parameter,
        value,
        unit,
        measured_at,
        latitude,
        longitude,
        row_number() over (
            partition by location_id, parameter, measured_at
            order by ingested_at desc
        ) as rn
    from source
    where value is not null
      and measured_at is not null
)

select
    location_id,
    location_name,
    parameter,
    value,
    unit,
    measured_at,
    latitude,
    longitude
from deduped
where rn = 1
