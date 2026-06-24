# Portugal Energy Grid Analytics

## Project summary

This project is an end-to-end data science prototype for analysing Portugal's
electricity distribution system. It forecasts national electricity consumption
and grid injection, evaluates forecast robustness across multiple horizons,
calculates an explainable operational risk proxy, and presents the results in
an interactive geographic dashboard.

The work combines E-REDES energy data, IPMA operational/recent weather, and
historical Open-Meteo reanalysis. It covers API validation, reproducible data
layers, exploratory analysis, course methods, chronological model evaluation,
direct multi-step forecasting, risk scoring, and presentation.

> The operational risk proxy is not a confirmed outage prediction or a direct
> probability of network failure. No labelled outage/failure target is
> available in this project.

## Problem and objectives

The project addresses four related questions:

1. How accurately can national electricity consumption be forecast?
2. How accurately can E-REDES grid injection be forecast, particularly beyond
   the next hour?
3. Can unusual operational conditions be summarized as an interpretable risk
   score when confirmed failure labels are unavailable?
4. Can the resulting data quality, weather, forecast, and risk information be
   reviewed through one dashboard?

## Architecture

```text
E-REDES API             IPMA APIs            Open-Meteo archive
    |              operational/recent       historical reanalysis
    |                       |                        |
    +-----------> raw data snapshots <---------------+
                            |
                         silver
                  cleaned canonical tables
                            |
                          gold
             15-minute and complete hourly index
                            |
            +---------------+----------------+
            |                                |
    calendar/lag features         2024-2025 weather alignment
            |                                |
            +----------> enriched gold <-----+
                            |
       EDA + course methods + chronological evaluation
                            |
          backtesting + direct multi-step forecasting
                            |
             operational risk proxy + reports
                            |
                  Streamlit dashboard
```

## Data sources

| Source | Role | Coverage used |
|---|---|---|
| E-REDES | Consumption, grid injection, and production context | 2024-01-01 to 2025-12-31 |
| IPMA | Operational/recent observations, warnings, and daily forecasts | Recent API window |
| Open-Meteo | Historical reanalysis for model training and enrichment | Hourly, 2024-2025 |

E-REDES datasets:

- `consumo-total-nacional`
- `energia-injetada-na-rede-de-distribuicao`
- `energia-produzida-total-nacional`

The public IPMA endpoints used here do not provide a complete hourly archive
for the E-REDES modelling window. Therefore, IPMA is retained for
operational/recent weather context, while historical Open-Meteo reanalysis is
used for 2024-2025 training and enrichment. These sources are not treated as
interchangeable.

Open-Meteo proxy locations: Lisboa, Porto, Coimbra, Faro, Evora, Viseu, and
Braganca.

## Pipeline

The processing flow is:

1. Validate the live E-REDES and IPMA APIs.
2. Extract bounded E-REDES snapshots into the raw layer.
3. Normalize schemas and types in the silver layer.
4. Build 15-minute gold datasets and a complete hourly index.
5. Create leakage-safe lag, rolling, calendar, and seasonal features.
6. Align historical Open-Meteo reanalysis with the 2024-2025 hourly window.
7. Run exploratory analysis and the required course methods.
8. Evaluate models using chronological holdouts and rolling-origin folds.
9. Train direct models for 1h, 6h, 24h, and 168h horizons.
10. Build the operational risk proxy and dashboard outputs.

Current modelling data:

| Dataset | Rows | Missing target rows |
|---|---:|---:|
| Consumption hourly | 17,544 | 25 |
| Grid injection hourly | 17,544 | 2 |
| Historical weather features | 17,544 | 0 |
| Consumption enriched | 17,544 | 25 |
| Grid injection enriched | 17,544 | 2 |

Missing targets remain in the complete hourly index to preserve exact temporal
spacing. They are excluded from model fitting.

## Models and analytical methods

Forecast models:

- seasonal naive baseline;
- Ridge regression;
- LASSO regression;
- Random Forest regression;
- Gradient Boosting regression.

Additional methods:

- correlation and Random Forest feature selection;
- IQR, z-score, and Isolation Forest outlier detection;
- principal component analysis;
- K-means and DBSCAN clustering;
- MAE, RMSE, zero-safe MAPE, and R-squared;
- bootstrap confidence intervals for MAE;
- chronological holdout evaluation;
- expanding-window rolling-origin backtesting.

Leakage controls:

- no random train/test split is used;
- rolling target features are shifted before aggregation;
- injection technology components are excluded from grid injection forecast
  features because the target is derived from those components;
- direct targets are shifted independently for each prediction horizon;
- validation scripts verify chronology and feature alignment.

## Main findings

### Rolling-origin backtesting

The most defensible model comparison uses 41 expanding-window folds with
168-hour test windows.

| Forecast target | Best model | MAE | RMSE | MAPE | R-squared |
|---|---|---:|---:|---:|---:|
| Consumption | Random Forest | 21,507.19 | 32,405.79 | 5.26% | 0.973 |
| Grid injection | LASSO | 56,252.94 | 75,944.84 | 6.87% | 0.954 |

Consumption Random Forest won 39 of 41 folds. Grid injection was less decisive:
LASSO and Random Forest each won 13 folds, Ridge won 10, and Gradient Boosting
won 5.

### Forecasting results by prediction horizon

| Forecast target | Horizon | Selected result | MAE | RMSE | R-squared |
|---|---:|---|---:|---:|---:|
| Consumption | 1h | Weather-enriched Random Forest | 34,623.88 | 51,690.22 | 0.965 |
| Consumption | 24h | Random Forest with lag 1 | 57,737.44 | 85,133.36 | 0.905 |
| Grid injection | 1h | Weather-enriched LASSO | 101,409.10 | 133,984.68 | 0.938 |
| Grid injection | 24h | Weather-enriched Ridge | 390,146.36 | 487,494.36 | 0.183 |

Main interpretation:

- consumption is more stable and remains useful at longer horizons;
- grid injection forecasts are credible primarily at short horizons;
- the 24h grid injection forecast has weak explanatory power;
- the 168h grid injection forecast is weak and exploratory;
- lag 1 is the dominant short-horizon signal, especially for grid injection;
- historical Open-Meteo reanalysis can improve selected scenarios but remains
  secondary to recent target history.

### Operational risk proxy

The risk score combines pressure, short-term change, seasonal deviation,
outlier indicators, and weather flags:

```text
risk_score = 100 * (
    0.30 * pressure_score
  + 0.25 * seasonal_deviation_score
  + 0.20 * change_score
  + 0.15 * outlier_score
  + 0.10 * weather_score
)
```

| Score | Interpretation |
|---:|---|
| 0-35 | Low |
| 35-60 | Medium |
| 60-80 | High |
| 80-100 | Critical |

This is an explainable operational risk proxy only. It must not be interpreted
as a direct probability of failure.

## Quick start

Supported Python: **3.11 or newer**. The final audit used Python 3.14.5.

From `project_energy_grid`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

python -m scripts.validate_pipeline
python -m scripts.validate_class_methods
python -m scripts.validate_backtesting
python -m scripts.validate_multistep
python -m scripts.validate_risk_score
python -m scripts.validate_dashboard
```

The validation commands require the ignored runtime data under `data/`. To
rebuild it, follow the complete pipeline below.

## Complete reproduction

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

python -m scripts.eda_report
python -m scripts.run_class_methods_analysis
python -m scripts.backtest_consumption
python -m scripts.backtest_injection
python -m scripts.robustness_checks

python -m scripts.extract_open_meteo_historical \
  --start-date 2024-01-01 \
  --end-date 2025-12-31

python -m scripts.build_weather_aligned_features
python -m scripts.build_gold_enriched
python -m scripts.multistep_consumption
python -m scripts.multistep_injection
python -m scripts.build_risk_score
python -m scripts.build_audit_metadata

python -m scripts.validate_pipeline
python -m scripts.validate_class_methods
python -m scripts.validate_backtesting
python -m scripts.validate_multistep
python -m scripts.validate_risk_score
python -m scripts.validate_dashboard
python -m compileall src scripts
git diff --check
```

## Reproducibility notes

- `requirements-lock.txt` pins the validated direct dependencies;
  `requirements.txt` is the unpinned convenience specification.
- Live APIs are mutable. A future rebuild can produce different rows and model
  metrics.
- Exact snapshot reproduction requires files matching the SHA-256 hashes in
  `reports/validation/data_provenance.json`.
- The original extraction did not persist API response timestamps. The
  provenance manifest therefore labels local raw-file modification times as
  the best available snapshot timestamps.
- `python -m scripts.build_audit_metadata` regenerates timestamp-quality and
  provenance reports after a data rebuild.
- Random seeds are fixed where supported, and temporal splits are deterministic.

Audit artifacts:

```text
reports/validation/data_provenance.json
reports/validation/timestamp_quality_report.csv
reports/validation/timestamp_quality_report.md
```

## Dashboard

Run locally from `project_energy_grid`:

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`. The dashboard provides separate consumption and
grid injection views for time series, forecasts, risk events, historical
weather, and data quality.

Recommended presentation order:

1. Executive overview.
2. Consumption forecast results.
3. Grid injection forecast results and long-horizon limitations.
4. Operational risk proxy events and methodology warning.
5. Historical Open-Meteo reanalysis alignment.
6. Data quality and DST limitations.

## Recommended review path

For a fast academic review:

1. Read this README and `SUBMISSION_NOTES.md`.
2. Review `reports/backtesting/model_selection_report.md` for robust model
   comparison.
3. Review `reports/multistep/multistep_model_report.md` for horizon-specific
   results.
4. Review `reports/risk/risk_score_report.md` for the proxy methodology.
5. Review `reports/validation/timestamp_quality_report.md` and
   `data_provenance.json` for data quality and reproducibility.
6. Inspect `src/models/`, `src/transform/`, and the orchestration scripts only
   after the reported methodology is clear.
7. Run the dashboard last as the presentation layer.

Files that matter most:

| Path | Purpose |
|---|---|
| `scripts/build_silver.py` | Raw-to-silver cleaning orchestration |
| `scripts/build_gold_hourly.py` | Hourly index and forecasting features |
| `src/models/evaluation.py` | Metrics and chronological holdout |
| `src/models/backtesting.py` | Expanding-window evaluation |
| `src/models/multistep.py` | Direct multi-horizon targets and features |
| `src/models/risk_score.py` | Operational risk proxy calculation |
| `src/dashboard/app.py` | Interactive presentation layer |

## Repository structure and Git hygiene

```text
project_energy_grid/
  data/                 # ignored runtime raw, silver, and gold datasets
  notebooks/            # API validation notebook
  reports/              # committed tables, figures, and audit reports
  scripts/              # extraction, pipeline, modelling, and validation entry points
  src/
    dashboard/          # Streamlit application
    extract/            # API clients
    models/             # modelling and evaluation helpers
    transform/          # cleaning and feature engineering
    utils/              # shared I/O
  README.md
  SUBMISSION_NOTES.md
  requirements.txt
  requirements-lock.txt
```

Do not commit runtime data or local environment artifacts. `.gitignore`
excludes:

- `data/raw/`, `data/silver/`, and `data/gold/`;
- all Parquet files;
- `.venv/`;
- `__pycache__/` and Python bytecode;
- `.ipynb_checkpoints/`;
- `.env`.

Reports, source code, scripts, notebooks, and documentation are eligible for
commit. Review `git status` and the staged file list before every submission.

## Known limitations

- No labelled outage/failure events are available; the risk score is not a
  failure probability.
- Grid injection forecasting is weak at 24h and exploratory at 168h.
- Direct multi-step results use one chronological 80/20 holdout; the full
  multi-step grid has not been evaluated with rolling-origin folds.
- Weather-enriched models use origin-time weather, not target-time weather
  forecasts.
- IPMA operational/recent weather does not cover the 2024-2025 modelling
  period; historical Open-Meteo reanalysis fills that analytical role.
- Open-Meteo is an auxiliary reanalysis source, not an official Portuguese
  station archive.
- E-REDES contains DST-correlated duplicate timestamps. They are retained
  because duplicate records contain different values; equal-weight hourly
  aggregation may bias four affected hours.
- Consumption has an additional unresolved 96-quarter-hour source gap across
  2025-10-13/14.
- The data window covers only two years and may not represent future structural
  changes.
- Live API rebuilds are not byte-for-byte reproducible without matching the
  recorded snapshot hashes.

## Google Colab

Colab can run validation, modelling, and report generation. Clone the
repository, enter the directory containing `src/` and `scripts/`, and install
the pinned requirements:

```bash
!git clone https://github.com/Usuario6/EDAII.git
%cd EDAII/Project/project_energy_grid
!pip install -r requirements-lock.txt
```

Because `data/` is ignored, either run the complete reproduction pipeline or
copy a trusted snapshot from Google Drive into `./data`. Do not hardcode a
GitHub token in a notebook. The Streamlit dashboard is more reliable for a
local presentation than through a Colab tunnel.
