# Feature Candidates

## A. Consumption model

- `total_lag_1`
- `total_lag_24`
- `total_lag_168`
- `total_rollmean_24`
- `total_rollmean_168`
- `hour`
- `dayofweek`
- `month`
- `is_weekend`
- temperature features, if aligned later

## B. Injection model

- `total_injection_lag_1`
- `total_injection_lag_24`
- `total_injection_lag_168`
- `total_injection_rollmean_24`
- `total_injection_rollmean_168`
- `hour`
- `dayofweek`
- `month`
- `cogeracao`
- `eolica`
- `fotovoltaica`
- `hidrica`
- `outras_tecnologias`
- `rede_dist`
- weather features, if aligned later

## C. Risk score

- normalized consumption
- normalized injection
- IPMA warnings
- heavy rain flag
- strong wind flag
- outlier indicators
- interruption indicators later
