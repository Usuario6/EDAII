# Portugal Energy Grid Analytics

Initial data-engineering foundation for a future predictive model of electricity consumption, distribution-grid injection, produced energy, and operational-risk proxies in Portugal, with geographic visualization planned for a later phase.

Current status:

- Live API validation works in the current environment.
- Bounded E-REDES historical extraction, cleaning, silver generation, gold generation, and EDA reporting are implemented.
- No machine-learning model or dashboard is implemented yet.

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
python -m scripts.validate_pipeline
python -m scripts.eda_report
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

## Next steps

- Start feature selection and baseline model design from the new gold datasets.
- Use IPMA alignment checks before adding weather joins into the modelling dataset.
