-- Air quality readings joined to the nearest weather forecast slot,
-- one row per station/parameter/hour. This is the table the dashboard reads.
--
-- BMKG publishes forecasts on a 3-hourly grid and only forward in time, so an
-- equality join on the truncated hour matches almost nothing: readings land on
-- arbitrary hours, and a reading taken now has no same-hour forecast row at
-- all. Pick the closest slot within +/- 3 hours (BMKG's own cadence) instead.
with air_quality as (
    select
        location_id,
        location_name,
        parameter,
        value,
        unit,
        date_trunc('hour', measured_at) as reading_hour,
        latitude,
        longitude
    from {{ ref('stg_air_quality') }}
),

weather as (
    select
        date_trunc('hour', forecast_at) as forecast_hour,
        avg(temperature_c) as temperature_c,
        avg(humidity_pct) as humidity_pct,
        avg(wind_speed_ms) as wind_speed_ms
    from {{ ref('stg_weather') }}
    group by 1
)

select
    aq.location_id,
    aq.location_name,
    aq.parameter,
    aq.value,
    aq.unit,
    aq.reading_hour,
    aq.latitude,
    aq.longitude,
    w.temperature_c,
    w.humidity_pct,
    w.wind_speed_ms,
    case
        when aq.parameter = 'pm25' and aq.value > 55 then 'unhealthy'
        when aq.parameter = 'pm25' and aq.value > 35 then 'moderate'
        else 'good'
    end as aqi_category
from air_quality aq
left join lateral (
    select temperature_c, humidity_pct, wind_speed_ms
    from weather
    where forecast_hour between aq.reading_hour - interval '3 hours'
                            and aq.reading_hour + interval '3 hours'
    order by abs(extract(epoch from forecast_hour - aq.reading_hour))
    limit 1
) w on true
