# Timestamp and DST Validation Report

This validation report documents duplicate and missing 15-minute timestamps without modifying the source snapshot.

## Validation Findings

All duplicated timestamps occur in the 01:00-01:45 UTC interval on Portugal's 2024 and 2025 daylight-saving transition dates. The raw E-REDES snapshot already contains repeated UTC values with different measurements, so the ambiguity is source-level and is not introduced by local timezone conversion.

Missing injection and production quarter-hours occur on the two autumn transition dates. Consumption has the same DST-related gaps plus a separate 96-quarter-hour gap spanning 2025-10-13/14; that additional gap is not a DST transition.

## Data-Handling Decision

No source record is silently dropped and no synthetic timezone offset is assigned. The existing hourly layer averages every record published for an hour. Hours with no source observations remain in the complete hourly index with missing targets and are excluded from model fitting. This conservative policy preserves traceability, but the four DST hours may be biased because repeated source records receive equal weight.

## Validation Results

```text
               dataset  rows  unique_timestamps  duplicated_timestamps_count  missing_quarter_hours_count                             duplicate_dates                               missing_dates
    silver_consumption 70088              70072                           16                          104 2024-03-31,2024-10-27,2025-03-30,2025-10-26 2024-10-27,2025-10-13,2025-10-14,2025-10-26
      silver_injection 70184              70168                           16                            8 2024-03-31,2024-10-27,2025-03-30,2025-10-26                       2024-10-27,2025-10-26
     silver_production 70184              70168                           16                            8 2024-03-31,2024-10-27,2025-03-30,2025-10-26                       2024-10-27,2025-10-26
gold_15min_consumption 70088              70072                           16                          104 2024-03-31,2024-10-27,2025-03-30,2025-10-26 2024-10-27,2025-10-13,2025-10-14,2025-10-26
  gold_15min_injection 70184              70168                           16                            8 2024-03-31,2024-10-27,2025-03-30,2025-10-26                       2024-10-27,2025-10-26
```

Machine-readable details and the decision are in `timestamp_quality_report.csv`.

## Known Limitations

The source does not provide enough timezone metadata to reconstruct the ambiguous DST records safely. Equal-weight hourly aggregation may bias the affected hours.
