# Submission Notes

## Project objective

Portugal Energy Grid Analytics is a data science prototype for forecasting
national electricity consumption and E-REDES grid injection, evaluating model
robustness across prediction horizons, calculating an explainable operational
risk proxy, and presenting the results in a Streamlit dashboard.

The operational risk proxy is not a confirmed outage prediction and must not be
interpreted as a direct probability of network failure. Labelled outage/failure
events are not available.

## Data sources

- **E-REDES:** national consumption, grid injection, and production context for
  2024-2025.
- **IPMA:** operational/recent observations, warnings, and forecasts.
- **Open-Meteo:** historical reanalysis aligned hourly with the 2024-2025
  E-REDES modelling window.

IPMA and Open-Meteo serve different purposes. Historical Open-Meteo reanalysis
supports model training and enrichment because the public IPMA endpoints used
in the project do not provide a complete hourly archive for 2024-2025.

## Pipeline summary

```text
API validation -> raw -> silver -> gold -> hourly index
-> calendar/lag/weather enrichment -> EDA and course methods
-> rolling-origin backtesting -> direct multi-step forecasts
-> operational risk proxy -> dashboard
```

The hourly datasets contain 17,544 timestamps. Missing targets remain in the
complete index to preserve chronology and are excluded from model fitting.
Leakage controls include chronological splits, shifted rolling statistics, and
exclusion of injection components from grid injection forecast features.

## Main results

Rolling-origin backtesting is the strongest model evidence:

| Target | Best model | MAE | RMSE | MAPE | R-squared |
|---|---|---:|---:|---:|---:|
| Consumption | Random Forest | 21,507.19 | 32,405.79 | 5.26% | 0.973 |
| Grid injection | LASSO | 56,252.94 | 75,944.84 | 6.87% | 0.954 |

Direct multi-step results show that consumption remains comparatively stable.
Grid injection performs well at 1h, degrades substantially at 6h, has weak
explanatory power at 24h, and is exploratory at 168h. Lag 1 is particularly
important for short-horizon grid injection forecasts.

## Known limitations

- The risk score is an operational proxy, not a failure probability.
- Direct multi-step results use one chronological 80/20 holdout rather than
  rolling-origin evaluation of the complete scenario grid.
- Weather-enriched models use origin-time weather, not target-time forecasts.
- Historical Open-Meteo reanalysis is an auxiliary source, not official IPMA
  station history.
- Source-level DST duplicates are retained because repeated timestamps contain
  different measurements; four hourly aggregates may therefore be biased.
- Consumption contains a separate 96-quarter-hour source gap in October 2025.
- Live API rebuilds may differ from the audited snapshot and its reported
  metrics.

## Reproduction

From `project_energy_grid`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Run the complete pipeline documented in `README.md`, or restore a trusted data
snapshot matching `reports/validation/data_provenance.json`. Then validate:

```bash
python -m scripts.validate_pipeline
python -m scripts.validate_class_methods
python -m scripts.validate_backtesting
python -m scripts.validate_multistep
python -m scripts.validate_risk_score
python -m scripts.validate_dashboard
python -m compileall src scripts
```

## Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`. Review consumption and grid injection separately,
then inspect forecast horizons, risk events, historical weather, and data
quality.

## Claims to avoid

Do not claim that the project predicts real failures, produces a calibrated
failure probability, proves operational performance outside 2024-2025, or
provides reliable long-horizon grid injection forecasts. The defensible claim
is that it provides validated short-horizon forecasting, exploratory
long-horizon analysis, and an explainable operational risk proxy.
