# Portugal Energy Grid Analytics

End-to-end EDA-II project for forecasting electricity consumption and distribution-grid injection in Portugal using E-REDES energy data, IPMA official weather context, Open-Meteo historical weather alignment, machine-learning models, operational-risk proxy scoring, and an interactive Streamlit dashboard.

The project follows a reproducible analytical pipeline:

```text
API validation → raw layer → silver layer → gold layer → hourly aggregation
→ weather alignment → feature enrichment → EDA/course methods
→ backtesting → multi-step forecasting → risk scoring → dashboard
```

## Current status

Completed:

- E-REDES and IPMA live API validation.
- E-REDES bounded historical extraction for 2024–2025.
- Raw, silver, gold, hourly and enriched data layers.
- Exploratory data analysis reports and plots.
- EDA-II course methods:
  - feature selection;
  - outlier detection;
  - PCA;
  - clustering;
  - regularized regression;
  - ensemble regression.
- Rolling-origin backtesting.
- Lag-dominance robustness checks.
- Direct multi-step forecasting for 1h, 6h, 24h and 168h.
- Historical hourly weather alignment using Open-Meteo reanalysis data.
- Weather-enriched forecasting scenario.
- Operational-risk proxy scoring.
- Interactive Streamlit dashboard.
- Validation scripts for pipeline, class methods, backtesting, multi-step forecasting, risk scoring and dashboard.
- Google Colab usage instructions for cloning, rebuilding data, or using cached data from Google Drive.

Important limitation:

> The project does **not** predict confirmed grid failures because no labelled outage or failure target is available. The operational-risk score is an explainable proxy based on pressure, deviations, outliers and weather flags.

## Data sources

### E-REDES

Official source for Portuguese national energy data:

- `consumo-total-nacional`
- `energia-injetada-na-rede-de-distribuicao`
- `energia-produzida-total-nacional`

### IPMA

Official Portuguese meteorological source used for:

- recent station observations;
- weather warnings;
- daily forecasts.

The public IPMA endpoint used in this project provides recent observations and warnings, but it does not expose a complete hourly historical archive aligned with the 2024–2025 E-REDES modelling window.

### Open-Meteo

Auxiliary historical weather source used only for 2024–2025 hourly weather alignment.

Open-Meteo historical reanalysis data is used to fill the historical weather gap and produce weather features aligned with the E-REDES hourly datasets.

Representative Portuguese locations used:

- Lisboa
- Porto
- Coimbra
- Faro
- Évora
- Viseu
- Bragança

## Data coverage

### E-REDES historical window

```text
2024-01-01 00:00 → 2025-12-31 23:45
```

### Hourly modelling window

```text
2024-01-01 00:00 → 2025-12-31 23:00
```

### Current row counts

| Layer | Dataset | Rows |
|---|---|---:|
| Raw | E-REDES consumption | 70,088 |
| Raw | E-REDES grid injection | 70,184 |
| Raw | E-REDES production | 70,184 |
| Silver | consumption | 70,088 |
| Silver | injection | 70,184 |
| Silver | production | 70,184 |
| Gold | consumption 15-min | 70,088 |
| Gold | injection 15-min | 70,184 |
| Gold | consumption hourly | 17,544 |
| Gold | injection hourly | 17,544 |
| Gold | consumption enriched | 17,544 |
| Gold | injection enriched | 17,544 |
| Gold | historical weather features | 17,544 |

### Missing hourly targets

| Dataset | Missing target rows | Missing target percentage |
|---|---:|---:|
| Consumption hourly | 25 | 0.1425% |
| Injection hourly | 2 | 0.0114% |

These rows are retained in the complete hourly index so that lag features remain chronologically exact. Model fitting excludes rows with missing targets.

## Weather alignment

Initial IPMA observations only covered a recent 2026 window and did not overlap with the 2024–2025 E-REDES modelling period.

The final project therefore uses Open-Meteo historical reanalysis for hourly weather enrichment.

Final weather alignment:

| Metric | Value |
|---|---:|
| Open-Meteo raw rows | 122,808 |
| Locations | 7 |
| Aggregated hourly weather rows | 17,544 |
| Duplicate weather timestamps | 0 |
| Overlap with consumption hourly | 17,544 |
| Overlap with injection hourly | 17,544 |
| Usable for historical modelling | True |
| Strong-wind hours | 435 |
| Heavy-rain hours | 1 |

Generated files:

```text
data/raw/open_meteo/open_meteo_historical_hourly.parquet
data/gold/gold_weather_features_hourly.parquet
reports/weather/weather_alignment_summary.csv
reports/weather/open_meteo_alignment_summary.csv
```

## Observed API schemas

### E-REDES consumption

```text
datahora, dia, mes, ano, date, time, bt, mt, at, mat, total
```

### E-REDES grid injection

```text
datahora, dia, mes, ano, date, time, cogeracao, eolica, fotovoltaica,
hidrica, outras_tecnologias, rede_dist
```

### E-REDES production

```text
datahora, dia, mes, ano, date, time, dgm, pre, total
```

### IPMA observations

```text
observation_time, station_id, intensidadeVentoKM, temperatura, radiacao,
idDireccVento, precAcumulada, intensidadeVento, humidade, pressao, value
```

### IPMA warnings

```text
text, awarenessTypeName, idAreaAviso, startTime, awarenessLevelID, endTime
```

### IPMA daily forecasts

```text
forecast_day, precipitaProb, tMin, tMax, predWindDir, idWeatherType,
classWindSpeed, longitude, globalIdLocal, latitude, classPrecInt
```

### Open-Meteo historical weather

```text
time, temperature_2m, relative_humidity_2m, precipitation, rain,
wind_speed_10m, wind_gusts_10m, shortwave_radiation, surface_pressure,
cloud_cover, location, latitude, longitude, datetime
```

## Setup

Supported Python: **3.11 or newer**. The final audit was executed with Python
3.14.5. For the closest dependency reproduction, install the pinned direct
dependencies from `requirements-lock.txt`; `requirements.txt` remains the
unpinned convenience specification.

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Live APIs are mutable. Rebuilding at a later date can produce different rows
and metrics. Exact snapshot reproduction requires data files matching the
SHA-256 hashes in `reports/validation/data_provenance.json`.

## Full pipeline execution

Run this sequence to rebuild the project from the source APIs and scripts:

```bash
python -m scripts.validate_apis

python -m scripts.extract_e_redes_window \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --page-size 100 \
  --max-pages 1000

python -m scripts.build_silver
python -m scripts.build_gold
python -m scripts.build_gold_hourly
python -m scripts.build_audit_metadata
python -m scripts.validate_pipeline

python -m scripts.eda_report
python -m scripts.run_class_methods_analysis
python -m scripts.validate_class_methods

python -m scripts.backtest_consumption
python -m scripts.backtest_injection
python -m scripts.robustness_checks
python -m scripts.validate_backtesting

python -m scripts.extract_open_meteo_historical \
  --start-date 2024-01-01 \
  --end-date 2025-12-31

python -m scripts.build_weather_aligned_features
python -m scripts.build_gold_enriched

python -m scripts.multistep_consumption
python -m scripts.multistep_injection
python -m scripts.validate_multistep

python -m scripts.build_risk_score
python -m scripts.validate_risk_score

python -m scripts.validate_dashboard
```

## Final validation command

Before submission or presentation, run:

```bash
python -m scripts.validate_pipeline
python -m scripts.validate_class_methods
python -m scripts.validate_backtesting
python -m scripts.validate_multistep
python -m scripts.validate_risk_score
python -m scripts.validate_dashboard
python -m compileall src scripts
```

Latest validation status:

```text
validate_class_methods: passed
validate_backtesting: passed
validate_multistep: passed
validate_risk_score: passed
validate_dashboard: passed
compileall: OK
```

## Course methods implemented

The project follows the CRISP-DM structure:

1. Business understanding
2. Data understanding
3. Data preparation
4. Modelling
5. Evaluation
6. Deployment/dashboard prototype

Implemented methods:

- MAE, RMSE, zero-safe MAPE and \(R^2\)
- Bootstrap confidence intervals for MAE
- Ridge regression
- LASSO regression
- Random Forest regression
- Gradient Boosting regression
- Correlation-based feature selection
- Random Forest feature importance
- IQR outlier detection
- z-score outlier detection
- Isolation Forest
- PCA
- K-means
- DBSCAN
- Silhouette evaluation
- Chronological train/test splitting
- Rolling-origin backtesting

Target leakage controls:

- Injection technology components are excluded from forecasting because `total_injection` is derived from them.
- Rolling means are shifted before calculation.
- No random train/test split is used.
- Validation scripts check chronological splits and leakage-safe features.

## Rolling-origin backtesting

Backtesting uses an expanding chronological training window.

Configuration:

| Parameter | Value |
|---|---:|
| Initial window | 60% of usable observations |
| Test window | 168 hours |
| Step | 168 hours |
| Folds | 41 per target |
| Split type | chronological expanding window |

Average backtesting results:

| Target | Best candidate | MAE | RMSE | MAPE | \(R^2\) |
|---|---|---:|---:|---:|---:|
| Consumption | Random Forest | 21,507.19 | 32,405.79 | 5.26% | 0.973 |
| Injection | LASSO | 56,252.94 | 75,944.84 | 6.87% | 0.954 |

Fold wins:

| Target | Result |
|---|---|
| Consumption | Random Forest won 39 of 41 folds |
| Injection | LASSO and Random Forest won 13 folds each; Ridge 10; Gradient Boosting 5 |

Main robustness finding:

Removing lag 1 substantially worsens performance. This confirms that the strongest signal is the most recent observation, especially for injection.

Outputs:

```text
reports/backtesting/
```

## Multi-step forecasting

Direct forecasting is implemented for:

```text
1 hour, 6 hours, 24 hours, 168 hours
```

Each horizon creates a separate shifted target. Every model uses only information known at forecast origin plus calendar features for the future timestamp.

Scenarios:

- with lag-1;
- without lag-1;
- calendar and seasonal features;
- seasonal naive baseline;
- weather enriched.

Selected results:

| Target | Horizon | Candidate | MAE | RMSE | \(R^2\) |
|---|---:|---|---:|---:|---:|
| Consumption | 1h | Weather-enriched Random Forest | 34,623.88 | 51,690.22 | 0.965 |
| Consumption | 24h | Random Forest with lag-1 | 57,737.44 | 85,133.36 | 0.905 |
| Injection | 1h | Weather-enriched LASSO | 101,409.10 | 133,984.68 | 0.938 |
| Injection | 24h | Weather-enriched Ridge | 390,146.36 | 487,494.36 | 0.183 |

Interpretation:

- Weather enrichment improves some horizons but does not dominate all cases.
- Lag features remain the strongest predictors.
- Consumption is more stable and easier to forecast than injection.
- Injection degrades strongly at longer horizons because it is more volatile and depends on external production conditions.
- Injection results are credible primarily at short horizons. The 24h result
  has weak explanatory power, and the 168h result is weak and exploratory; do
  not present either as established operational performance.

Outputs:

```text
reports/multistep/consumption_multistep_results.csv
reports/multistep/injection_multistep_results.csv
reports/multistep/consumption_multistep_summary.csv
reports/multistep/injection_multistep_summary.csv
reports/multistep/lag1_dependency_summary.csv
reports/multistep/multistep_model_report.md
```

## Operational-risk proxy

The risk score is an explainable operational proxy, not a labelled failure probability.

The score combines:

- pressure score;
- change score;
- seasonal deviation score;
- outlier score;
- weather score.

Risk-score formula:

```text
risk_score = 100 * (
    0.30 * pressure_score
  + 0.25 * seasonal_deviation_score
  + 0.20 * change_score
  + 0.15 * outlier_score
  + 0.10 * weather_score
)
```

Risk levels:

| Score | Level |
|---:|---|
| 0–35 | low |
| 35–60 | medium |
| 60–80 | high |
| 80–100 | critical |

Current risk-score summary:

| Dataset | Rows | Scored rows | Mean risk | Max risk | High | Critical | Weather-flag hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Consumption | 17,544 | 17,519 | 26.65 | 90.0 | 395 | 11 | 435 |
| Injection | 17,544 | 17,542 | 18.24 | 100.0 | 304 | 66 | 435 |

Outputs:

```text
reports/risk/consumption_risk_score.csv
reports/risk/injection_risk_score.csv
reports/risk/risk_score_summary.csv
reports/risk/risk_score_report.md
reports/risk/consumption_risk_score_plot.png
reports/risk/injection_risk_score_plot.png
```

## Dashboard

The project includes an interactive Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

Open locally:

```text
http://localhost:8501
```

Dashboard sections:

- Executive overview
- Time series
- Risk events
- Model comparison
- Weather
- Data quality
- Methodology

Recommended presentation order:

1. Executive overview
2. Time series using daily mean
3. Model comparison
4. Risk events
5. Weather alignment
6. Data quality
7. Methodology disclaimer

## Google Colab usage

The repository can be cloned and executed in Google Colab, but only the code and committed reports are available through Git.

The runtime datasets under `data/` are ignored by Git. Therefore, in Colab the data must either be:

1. rebuilt from the APIs; or
2. copied from Google Drive if the local `data/` folder was previously uploaded there.

### Clone the repository in Colab

For a public repository:

```bash
!git clone https://github.com/Usuario6/EDAII.git
%cd EDAII/Project/project_energy_grid
!pip install -r requirements.txt
```

If the folder path is uncertain, locate it with:

```bash
!find . -maxdepth 5 -name "validate_pipeline.py"
!find . -maxdepth 5 -name "README.md"
```

Then move into the folder that contains `scripts/`, `src/` and `requirements.txt`.

### Clone a private repository

If the GitHub repository is private, create a temporary GitHub token with repository read access and run:

```python
from getpass import getpass
token = getpass("GitHub token: ")
```

```bash
!git clone https://$token@github.com/Usuario6/EDAII.git
%cd EDAII/Project/project_energy_grid
!pip install -r requirements.txt
```

Do not hardcode or commit the token.

### Option A: rebuild the data in Colab

```bash
!python -m scripts.validate_apis

!python -m scripts.extract_e_redes_window \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --page-size 100 \
  --max-pages 1000

!python -m scripts.build_silver
!python -m scripts.build_gold
!python -m scripts.build_gold_hourly
!python -m scripts.validate_pipeline

!python -m scripts.extract_open_meteo_historical \
  --start-date 2024-01-01 \
  --end-date 2025-12-31

!python -m scripts.build_weather_aligned_features
!python -m scripts.build_gold_enriched

!python -m scripts.multistep_consumption
!python -m scripts.multistep_injection
!python -m scripts.validate_multistep

!python -m scripts.build_risk_score
!python -m scripts.validate_risk_score
```

### Option B: use Google Drive for cached data

If the `data/` folder was uploaded to Google Drive, mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy the cached data into the cloned repository:

```bash
!cp -r /content/drive/MyDrive/project_energy_grid_data/data ./data
```

Then validate:

```bash
!python -m scripts.validate_pipeline
!python -m scripts.validate_multistep
!python -m scripts.validate_risk_score
```

### Dashboard note for Colab

The Streamlit dashboard is intended to run locally:

```bash
streamlit run src/dashboard/app.py
```

Colab can be used for pipeline validation, model execution and report generation, but the dashboard is easier and more reliable to demonstrate from a local machine.

### Colab compatibility checklist

- Use only relative paths inside scripts.
- Do not hardcode local paths such as `/mnt/shared/...`.
- Rebuild `data/` from APIs or copy it from Google Drive.
- Run scripts with `python -m scripts.script_name`.
- Confirm `requirements.txt` includes all required packages, including `pandas`, `scikit-learn`, `pyarrow`, `requests`, `streamlit` and `plotly`.

## Repository structure

```text
data/
  raw/
  silver/
  gold/

notebooks/
  01_api_validation.ipynb

reports/
  backtesting/
  clustering/
  dimensionality/
  eda/
  models/
  multistep/
  outliers/
  risk/
  validation/
  weather/

scripts/
  extract_e_redes_window.py
  extract_open_meteo_historical.py
  build_silver.py
  build_gold.py
  build_gold_hourly.py
  build_weather_aligned_features.py
  build_gold_enriched.py
  run_class_methods_analysis.py
  backtest_consumption.py
  backtest_injection.py
  robustness_checks.py
  multistep_consumption.py
  multistep_injection.py
  build_risk_score.py
  validate_pipeline.py
  validate_class_methods.py
  validate_backtesting.py
  validate_multistep.py
  validate_risk_score.py
  validate_dashboard.py

src/
  dashboard/
  extract/
  models/
  transform/
  utils/
```

Runtime datasets under `data/` are excluded from Git.

## Data provenance and timestamp quality

Run the auditable metadata generator after rebuilding the data:

```bash
python -m scripts.build_audit_metadata
```

Generated audit artifacts:

```text
reports/validation/timestamp_quality_report.csv
reports/validation/timestamp_quality_report.md
reports/validation/data_provenance.json
```

The provenance manifest records source APIs, available snapshot timestamps,
row counts, date ranges and SHA-256 hashes. The original extraction scripts did
not persist API response timestamps, so raw-file modification times are clearly
identified as the best available snapshot timestamp rather than an exact server
extraction timestamp.

The duplicated 15-minute timestamps coincide with Portuguese DST transitions
and already exist in the E-REDES source snapshot with different values. They
are retained for traceability rather than silently dropped or assigned invented
offsets. The hourly layer averages all published records in each hour. Missing
hours remain missing in the complete hourly index and are excluded from model
fitting. See the timestamp report for counts, dates and the remaining bias risk.

## Known limitations

- The operational-risk score is a proxy, not a real grid-failure probability.
- There are no labelled outage/failure events.
- IPMA public observations are recent and not historically aligned with 2024–2025.
- Open-Meteo reanalysis is used as an auxiliary historical weather source, not as an official Portuguese station archive.
- Lag-1 remains the dominant predictor, especially for short-horizon forecasts.
- Injection is harder to forecast at longer horizons because it depends strongly on generation conditions.
- Raw and 15-minute E-REDES layers contain 16 duplicated timestamps, but the hourly modelling layers have zero duplicated timestamps.
- The complete hourly index keeps 25 missing consumption targets and 2 missing injection targets to preserve exact hourly chronology.
- The DST-correlated duplicate records are retained. Equal-weight hourly
  aggregation may bias the four affected hours, and the separate October 2025
  consumption gap remains unresolved in the source snapshot.
- Live API rebuilds are not byte-for-byte reproducible without the snapshot
  files matching `reports/validation/data_provenance.json`.

## Final project conclusion

The final project delivers a validated weather-aware forecasting and monitoring prototype for Portugal’s electricity system.

It combines official E-REDES energy data, IPMA weather context, Open-Meteo historical weather enrichment, machine-learning forecasting, operational-risk proxy scoring and an interactive dashboard.

The strongest models achieve high short-horizon performance, especially for consumption. Weather enrichment is useful but secondary to lag-based signals. The dashboard provides an interpretable operational layer for exploring consumption, injection, model performance, weather alignment and high-risk periods.
