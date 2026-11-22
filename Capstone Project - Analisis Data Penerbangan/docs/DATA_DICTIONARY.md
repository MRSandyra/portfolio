# Data Dictionary

The cleaned dataset uses one row per unique, valid **flight offer listing**. It does not contain bookings or ticket sales.

| Source column | Clean column | Type | Meaning | Validation / caveat |
|---|---|---:|---|---|
| `Airline name ` | `airline` | string | Airline label shown in the listing | Missing labels and observed promotion text are rejected |
| `Travel  Time` | `travel_time` | string | Source duration label | Retained for traceability |
| `Travel  Time` | `travel_minutes` | integer | Parsed total travel duration | Must be positive; formats such as `21h 15m` |
| `Number of  Stoppage ` | `stops` | string | Source stop label | Retained for traceability |
| `Number of  Stoppage ` | `stop_count` | integer | Parsed number of stops | Non-negative; `Nonstop` becomes `0` |
| `Ticket prize(Doller)` | `fare_usd` | float | Listed fare, assumed USD from the source header | Must be numeric and positive; currency was not independently verified |
| `Depreture  Airport` | `departure_airport` | string | Departure airport code | Uppercased; missing values retained and reported |
| `Depreture Time` | `departure_time` | string | Source departure clock label | No date or time zone is supplied |
| `Depreture Time` | `departure_minutes` | nullable integer | Minutes after local midnight | Invalid/missing clock labels remain missing |
| `1st Stoppage Waiting Hour` | `first_stop_wait` | string | First connection wait label | Optional; not used for core KPIs |
| `1st Stoppage` | `first_stop_airport` | string | First connection airport | Optional source field |
| `Destination Airport` | `destination_airport` | string | Destination airport code | Uppercased; missing values retained and reported |
| derived | `route` | nullable string | `departure_airport-destination_airport` | Created only when both codes exist |
| `Arrival Time` | `arrival_time` | string | Source arrival clock label | No time zone is supplied |
| `Arrival Time` | `arrival_minutes` | nullable integer | Minutes after local midnight | Invalid/missing clock labels remain missing |
| `Arrival Date` | `arrival_date` | string | Yearless label such as `Apr 8` | Not a complete calendar date |
| `Arrival Date` | `arrival_month`, `arrival_day` | nullable integer | Parsed month/day | Validated against leap year 2000; no year is inferred |
| derived | `fare_per_travel_hour_usd` | float | Fare divided by scheduled travel hours | Descriptive only; not profitability |
| derived | `is_nonstop` | boolean | Whether `stop_count == 0` | Based on source label |
| derived | `source_row` | integer | Original CSV line number including header offset | Supports rejected-row audit |

Rows are excluded only for an exact duplicate, missing/suspect airline, invalid/non-positive fare, invalid/non-positive travel duration, or unparseable stop count. Missing route or arrival date is retained and reported separately.

