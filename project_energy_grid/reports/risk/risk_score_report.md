# Operational risk proxy score report

## Purpose

This report creates an operational risk proxy score for electricity consumption and energy injection.

The score does not represent a real probability of grid failure because the current project has no labelled failure events.

Instead, the score combines observable indicators:

- current pressure level;
- abnormal change from recent and seasonal references;
- deviation from rolling behaviour;
- outlier detection;
- weather flags, only where historically available.

## Formula logic

The score is scaled from 0 to 100:

```text
risk_score = 100 * (
    0.30 * pressure_score
  + 0.25 * seasonal_deviation_score
  + 0.20 * change_score
  + 0.15 * outlier_score
  + 0.10 * weather_score
)
```

## Risk levels

| Score | Level |
|---:|---|
| 0–35 | low |
| 35–60 | medium |
| 60–80 | high |
| 80–100 | critical |

## Summary

```text
    dataset  rows  scored_rows  missing_target_rows  mean_risk_score  max_risk_score  low_count  medium_count  high_count  critical_count  missing_target_count  outlier_count  weather_flag_count
consumption 17544        17519                   25        26.651526            90.0      14236          2877         395              11                    25            354                 435
  injection 17544        17542                    2        18.242133           100.0      15379          1793         304              66                     2            462                 435
```

## Interpretation

The risk score should be interpreted as an explainable operational pressure indicator, not as confirmed outage prediction.

High values identify timestamps where the system behaviour is unusual, elevated, or unstable relative to recent and historical patterns.

Weather contribution uses aligned Open-Meteo reanalysis for 2024–2025. IPMA observations remain current/recent operational context and are not force-filled into the historical modelling period.
