-- The +7h timezone bug (BMKG "local_datetime" inserted into a TIMESTAMPTZ
-- column) pushed the whole forecast window into the future and silently
-- emptied the weather join, while all ten column tests stayed green. BMKG's
-- first slot is always the current 3-hour bucket, so anything further out
-- than 6 hours means the timestamps are being read in the wrong timezone.
select min(forecast_at) as earliest_forecast
from {{ ref('stg_weather') }}
having min(forecast_at) > now() + interval '6 hours'
