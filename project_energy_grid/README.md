# Portugal Energy Grid Analytics

Initial data-engineering foundation for a future predictive model of electricity consumption, distribution-grid injection, produced energy, and operational-risk proxies in Portugal, with geographic visualization planned for a later phase.

Current status:

- Live API validation works in the current environment.
- Bounded E-REDES historical extraction, cleaning, silver generation, gold generation, and EDA reporting are implemented.
- Chronological hourly baselines and the EDL II course analysis methods are implemented.
- No final model or dashboard is implemented yet.

## Data sources

- E-REDES OpenDataSoft: `consumo-total-nacional`
- E-REDES OpenDataSoft: `energia-injetada-na-rede-de-distribuicao`
- E-REDES OpenDataSoft: `energia-produzida-total-nacional`
- IPMA hourly station observations
- IPMA weather warnings
- IPMA daily city forecasts for horizons 0, 1, and 2

The validation pipeline requests only five E-REDES records per dataset. It does not perform full-dataset downloads.

## Live validation status

Validated on 2026-06-23.

- E-REDES `consumo-total-nacional`: `200`, 5 rows, saved to `data/raw/e_redes/e_redes_consumption_sample.parquet`
- E-REDES `energia-injetada-na-rede-de-distribuicao`: `200`, 5 rows, saved to `data/raw/e_redes/e_redes_grid_injection_sample.parquet`
- E-REDES `energia-produzida-total-nacional`: `200`, 5 rows, saved to `data/raw/e_redes/e_redes_production_sample.parquet`
- IPMA observations: `200`, 5328 rows, saved to `data/raw/ipma/ipma_observations_sample.parquet`
- IPMA warnings: `200`, 214 rows, saved to `data/raw/ipma/ipma_warnings_sample.parquet`
- IPMA daily forecasts day 0, 1, 2: `200`, 27 rows each, saved to `data/raw/ipma/ipma_forecast_day_0_sample.parquet`, `..._1_...`, `..._2_...`

Bounded E-REDES historical extraction window:

- `2025-01-01` to `2025-03-31`
- Saved to `data/raw/e_redes/e_redes_consumption_window.parquet`
- Saved to `data/raw/e_redes/e_redes_grid_injection_window.parquet`
- Saved to `data/raw/e_redes/e_redes_production_window.parquet`

Current row counts:

- `silver/e_redes_consumption.parquet`: 8644
- `silver/e_redes_injection.parquet`: 8644
- `silver/e_redes_production.parquet`: 8644
- `gold/gold_consumption.parquet`: 8644
- `gold/gold_injection.parquet`: 8644
- `gold/gold_consumption_hourly.parquet`: 2160
- `gold/gold_injection_hourly.parquet`: 2160
- `gold/gold_weather_hourly.parquet`: 5328

## First execution

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.validate_apis
python -m scripts.extract_e_redes_window --start-date 2025-01-01 --end-date 2025-03-31 --page-size 100 --max-pages 100
python -m scripts.build_silver
python -m scripts.build_gold
python -m scripts.build_gold_hourly
python -m scripts.validate_pipeline
python -m scripts.eda_report
python -m scripts.run_class_methods_analysis
python -m scripts.validate_class_methods
```

The validation prints each HTTP status, shape, column list, and first rows, then writes CSV and Parquet samples under `data/raw/e_redes/` and `data/raw/ipma/`.

## Observed API schemas

The real column names detected during validation are documented here after the first live execution.

<!-- API_SCHEMA_START -->
Validated on 2026-06-21. Fields are reproduced exactly as returned by each API, except for the explicitly added `forecast_day`, `observation_time`, `station_id`, and fallback `value` fields used while flattening IPMA responses.

- E-REDES `consumo-total-nacional`: `datahora`, `dia`, `mes`, `ano`, `date`, `time`, `bt`, `mt`, `at`, `mat`, `total`
- E-REDES `energia-injetada-na-rede-de-distribuicao`: `datahora`, `dia`, `mes`, `ano`, `date`, `time`, `cogeracao`, `eolica`, `fotovoltaica`, `hidrica`, `outras_tecnologias`, `rede_dist`
- E-REDES `energia-produzida-total-nacional`: `datahora`, `dia`, `mes`, `ano`, `date`, `time`, `dgm`, `pre`, `total`
- IPMA observations: `observation_time`, `station_id`, `intensidadeVentoKM`, `temperatura`, `radiacao`, `idDireccVento`, `precAcumulada`, `intensidadeVento`, `humidade`, `pressao`, `value`
- IPMA warnings: `text`, `awarenessTypeName`, `idAreaAviso`, `startTime`, `awarenessLevelID`, `endTime`
- IPMA daily forecasts (days 0, 1, and 2): `forecast_day`, `precipitaProb`, `tMin`, `tMax`, `predWindDir`, `idWeatherType`, `classWindSpeed`, `longitude`, `globalIdLocal`, `latitude`, `classPrecInt`
<!-- API_SCHEMA_END -->

## Pipeline outputs

- Silver datasets are written to `data/silver/`.
- Gold datasets are written to `data/gold/`.
- `gold_weather_hourly.parquet` is optional and is only produced when the IPMA observation timestamps can be parsed cleanly.
- EDA outputs are written to `reports/eda/` as `.csv` summaries and `.png` plots.

## Known issues

- The previous container could not resolve the E-REDES and IPMA hosts during a live rerun; that issue is now resolved in the current environment.
- IPMA observations are a short, recent window and may not support stable long-horizon joins yet.
- Operational risk must be treated as a proxy score, not a real failure prediction.
- E-REDES lag features are sparse at the start of the historical window, which is expected.
- IPMA observations are from 2026 while the bounded E-REDES window is from 2025, so weather is not joined to the current modelling tables.

## Course methods implemented

The workflow follows CRISP-DM: business and data understanding are documented, data preparation produces silver and hourly gold tables, modelling compares reproducible candidates, and evaluation uses a chronological holdout.

- Model evaluation: MAE, RMSE, zero-safe MAPE, R2, and bootstrap confidence intervals for MAE.
- Regularization: Ridge and LASSO with standardized features and time-series cross-validation.
- Ensemble learning: Random Forest and Gradient Boosting regressors.
- Feature selection: correlation filtering and Random Forest importance.
- Outlier detection: IQR, z-score, and Isolation Forest.
- Dimensionality reduction: standardized PCA with component loadings and explained variance.
- Clustering: K-means and DBSCAN with silhouette evaluation.

The consumption and injection models use only calendar, lag, and prior-window rolling features. Injection technology components are excluded from prediction because `total_injection` is their sum and including them would be target leakage. Rolling means are shifted by one observation before calculation.

Run the complete analysis and validation:

```bash
python -m scripts.build_gold_hourly
python -m scripts.run_class_methods_analysis
python -m scripts.validate_class_methods
```

Outputs are stored under `reports/models/`, `reports/outliers/`, `reports/dimensionality/`, and `reports/clustering/`. Runtime datasets remain excluded from Git.

## Next steps

- Extend the bounded historical window and evaluate stability with rolling-origin backtesting.
- Acquire historically aligned IPMA observations before adding weather features.
- Select a candidate model only after comparing performance across multiple temporal folds.
