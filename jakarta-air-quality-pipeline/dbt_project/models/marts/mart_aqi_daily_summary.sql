-- One row per station/day: daily average, peak, and worst AQI category seen.
-- Powers the "30-day trend" chart on the dashboard.
select
    location_id,
    location_name,
    parameter,
    date_trunc('day', reading_hour) as reading_day,
    avg(value) as avg_value,
    max(value) as max_value,
    count(*) filter (where aqi_category = 'unhealthy') as unhealthy_hours
from {{ ref('mart_aqi_hourly') }}
group by 1, 2, 3, 4
