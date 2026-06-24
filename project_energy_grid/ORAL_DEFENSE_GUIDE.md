# Oral Defense Guide: Portugal Energy Grid Analytics

## 1. One-minute project summary

This project is an end-to-end data science prototype for Portugal's electricity
distribution context. It uses E-REDES data to forecast national electricity
consumption and grid injection. IPMA provides official operational/recent
weather context, while historical Open-Meteo reanalysis provides the hourly
2024-2025 weather coverage needed for model training and enrichment.

The pipeline validates and extracts APIs, builds raw, silver, gold, hourly, and
weather-enriched datasets, performs exploratory and course-method analysis,
trains several regression models, evaluates them chronologically, calculates an
operational risk proxy, and presents the results in a Streamlit dashboard.

The project predicts consumption and grid injection. It does **not** predict
confirmed failures because no labelled outage or failure target is available.
The strongest rolling-origin result is the consumption Random Forest, with mean
MAE 21,507.19, mean RMSE 32,405.79, mean MAPE 5.26%, and mean R-squared 0.973
across 41 folds. Grid injection is also strong at short horizons but becomes
weak at 24 hours and exploratory at 168 hours.

## 2. Project objective and scope

### Initial idea

The initial concept was broader: forecast electricity consumption, grid
injection, and operational risk using E-REDES, IPMA, and REN, then communicate
the results through a dashboard.

### Final implemented scope

The implemented project includes:

- E-REDES consumption and grid injection forecasting;
- E-REDES production as contextual data;
- IPMA operational/recent weather observations, warnings, and forecasts;
- historical Open-Meteo reanalysis aligned with 2024-2025;
- raw, silver, gold, hourly, and enriched data layers;
- exploratory analysis and EDL-II course methods;
- chronological holdouts and rolling-origin backtesting;
- direct 1h, 6h, 24h, and 168h forecasting;
- an explainable operational risk proxy;
- an interactive Streamlit dashboard.

### Why REN remained future work

REN could add national electricity-balance, generation, import, and export
context. It was not integrated because E-REDES already supplied the active
targets and the project already required substantial temporal, weather, and
modelling alignment. Adding REN would have expanded scope without resolving the
main forecasting questions. REN is therefore a justified future extension, not
part of the active implementation.

### Why risk became a proxy

The available data has no confirmed outage labels, failure causes, or verified
network-stress classes. A supervised failure model would therefore be
methodologically indefensible. The project instead constructs a transparent
score from observed pressure, changes, seasonal deviations, outliers, and
weather flags. It is a monitoring and interpretation layer, not failure
prediction.

**Defense sentence:** "I reduced the scope when the data did not support the
original claim. Forecasting uses observed targets; risk remains an explainable,
uncalibrated proxy."

## 3. Dataset selection and reasoning

| Source | Contribution | Why selected | Usage | Main limitation |
|---|---|---|---|---|
| E-REDES | National consumption, distribution-grid injection, and production context | It directly provides the measurable energy targets | Actively used for extraction, modelling, evaluation, and dashboard data | Source timestamps contain DST-related ambiguity and some missing intervals |
| IPMA | Recent station observations, warnings, and short-term forecasts | It is the official Portuguese meteorological context source | Actively extracted and used as operational/recent context | The public endpoints used do not provide a complete hourly 2024-2025 archive |
| Open-Meteo | Hourly temperature, humidity, precipitation, wind, radiation, pressure, and cloud reanalysis | It fills the historical weather-coverage gap for the energy modelling window | Actively used for 2024-2025 enrichment through seven Portuguese proxy locations | It is auxiliary reanalysis, not an official IPMA station archive or target-time weather forecast |
| REN Data Hub | Potential broader system balance, generation, import, and export context | It could improve national-system interpretation and exogenous forecasting | Considered only; not integrated | Additional extraction and temporal alignment would expand scope significantly |

### Essential distinction

- **E-REDES:** main energy targets.
- **IPMA:** official operational/recent weather context.
- **Historical Open-Meteo reanalysis:** 2024-2025 weather training and
  enrichment.
- **REN:** future extension, not active implementation.

## 4. Pipeline explanation

```text
APIs
  -> raw snapshots
  -> silver cleaned tables
  -> gold analytical tables
  -> complete hourly modelling index
  -> calendar and historical weather enrichment
  -> EDA, course techniques, models, and validation
  -> operational risk proxy and reports
  -> Streamlit dashboard
```

### How to explain each layer

1. **API extraction:** Validate source access and schemas, then perform bounded
   extraction for the chosen dates. E-REDES extraction uses date chunks,
   pagination, page-size bounds, retries, and ordered timestamps.
2. **Raw layer:** Preserve local API snapshots so transformations can be rerun
   without immediately calling the APIs again.
3. **Silver layer:** Normalize column names, parse timestamps, convert numeric
   fields, replace common missing markers, and remove exact duplicate rows.
   Ambiguous duplicate timestamps are retained rather than silently discarded.
4. **Gold layer:** Create analytical consumption and grid injection targets and
   add time, lag, and rolling features at 15-minute resolution.
5. **Hourly layer:** Resample to a complete hourly index. This aligns energy and
   weather, lowers computational cost, and makes daily and weekly lags easier to
   interpret.
6. **Enriched layer:** Add Portuguese civil-time calendar features and merge the
   historical Open-Meteo reanalysis table one-to-one by UTC timestamp.
7. **Reports:** Store EDA, model comparisons, outliers, PCA, clustering,
   backtesting, multistep, weather, risk, and validation outputs.
8. **Dashboard:** Read enriched data and committed reports to provide a single
   interactive interpretation layer.

**Important validation wording:** `validate_pipeline` is an observational
inspection script. It prints schemas, coverage, duplicates, missing values, and
numeric summaries; it is not a strict assertion-based validator for every data
quality condition.

## 5. Feature engineering

### Time and calendar features

- hour, day of week, and month;
- weekend, workday, and Portuguese national holiday flags;
- day of year, week of year, quarter, and season;
- month-start and month-end indicators;
- cyclical hour and day-of-year encodings.

UTC API timestamps are converted to `Europe/Lisbon` for Portuguese calendar
semantics.

### Lag features

- **Lag 1:** most recent previous hourly observation;
- **Lag 24:** corresponding observation from the previous day;
- **Lag 168:** corresponding observation from the previous week.

These encode persistence and recurring daily/weekly patterns.

### Rolling features

The project uses 24-hour and 168-hour rolling means. The target is shifted by
one row **before** the rolling mean is calculated. Therefore, the current target
cannot enter its own predictor.

### Weather-derived features

- national proxy temperature, humidity, precipitation, rain, wind, gusts,
  radiation, pressure, and cloud cover;
- heating-degree and cooling-degree indicators;
- heavy-rain and strong-wind flags.

### Leakage prevention

- timestamps are sorted chronologically before lag creation;
- lags use only positive backward shifts;
- rolling statistics shift before aggregation;
- no random train/test split is used;
- forecast-calendar features refer to the future timestamp but are known in
  advance;
- direct targets are independently shifted for each horizon;
- injection component variables are excluded from grid injection forecast
  features because the target is derived from those components.

## 6. Models used

| Model | Why it was used | Contribution and result |
|---|---|---|
| Seasonal naive | Establishes a simple daily or weekly reference that advanced models should beat | Easy to interpret but substantially weaker than trained models, especially for grid injection |
| Ridge | Stabilizes correlated linear predictors through L2 regularization | Strong and stable linear candidate; nearly tied with LASSO for short-horizon injection |
| LASSO | Uses L1 regularization and can shrink weak coefficients toward zero | Best grid injection model by mean rolling-origin RMSE; weakens strongly without lag 1 |
| Random Forest | Captures nonlinear relationships and feature interactions without assuming a linear form | Best consumption model in 39 of 41 rolling-origin folds; weaker and less decisive for injection |
| Gradient Boosting | Builds sequential trees that correct earlier errors | Competitive for consumption and best 6h grid injection direct result, but not the overall rolling-origin winner |

Do not say a model is universally best. The winner depends on target, horizon,
scenario, and metric.

## 7. Course techniques and slide mapping

The page numbers below refer to PDF pages in the provided course decks. Where
the printed slide number differs, both are shown.

| Report section | Technique | Course deck / slide or page | Project evidence | How to explain orally |
|---|---|---|---|---|
| Sections 1 and 21 | CRISP-DM | `01 _General-Fundamentals_1.pdf`, PDF pp. 3-4 | README and complete pipeline | "The project moves from business understanding and data preparation to modelling, evaluation, and a dashboard prototype." |
| Sections 22-25 | Model evaluation | Project-specific regression application; no exact MAE/RMSE slide found | `src/models/evaluation.py` | "I used complementary absolute, squared, percentage, and variance-explanation metrics." |
| Sections 19 and 24 | Chronological validation | `05-Sl_AvaliacaoSelecaoModelos.pdf`, validation topic, PDF pp. 3-10 | `time_series_train_test_split` | "I adapted validation to preserve time order rather than applying shuffled K-fold." |
| Section 24 | Rolling-origin backtesting | Project-specific time-series extension of model validation | `src/models/backtesting.py`, `reports/backtesting/` | "The origin advances weekly while the training window expands, simulating repeated forecasts." |
| Sections 20-21 | Regularization | `05-Sl_AvaliacaoSelecaoModelos.pdf`, PDF pp. 38-42, printed slides 35-38 | `src/models/regularized.py` | "Regularization controls coefficient magnitude and reduces overfitting with correlated lag features." |
| Section 20.2 | Ridge | Same deck, PDF p. 40, printed slide 37 | `train_ridge_model` | "Ridge retains correlated predictors but shrinks their coefficients." |
| Section 20.2 | LASSO | Same deck, PDF p. 41, printed slide 38 | `train_lasso_model` | "LASSO uses an L1 penalty and can suppress weak coefficients." |
| Sections 20.3 and 25 | Random Forest | `04-PT_Sl_TecnicasMelhoria.pdf`, PDF pp. 12-15 | `src/models/ensemble.py` | "It combines many trees and captures nonlinear temporal interactions." |
| Sections 20.3 and 27 | Gradient Boosting | Same deck, boosting topic, PDF pp. 16-20 | `src/models/ensemble.py` | "Sequential trees focus on correcting previous residual errors." |
| Section 21 | Feature selection | `05-Sl_AvaliacaoSelecaoModelos.pdf`, PDF pp. 32-37, printed slides 29-34 | `src/models/feature_selection.py` | "I compared target correlation with model-based Random Forest importance." |
| Sections 21 and 31 | Outlier detection | `06-Sl_DetecaoOutliers.pdf`, PDF pp. 3-6 | `src/models/outliers.py` | "I combined simple statistical rules with a multivariate detector." |
| Section 31 | IQR | Same deck, PDF p. 11 | `detect_iqr_outliers` | "Values outside 1.5 IQR beyond the quartiles are flagged." |
| Section 31 | z-score | Same deck, PDF p. 12 | `detect_zscore_outliers` | "Values more than three standard deviations from the mean are flagged." |
| Section 31 | Isolation Forest | Outlier lecture topic; no exact Isolation Forest slide found | `detect_isolation_forest_outliers` | "This is a project extension for multivariate anomaly detection with fixed 2% contamination." |
| Section 31 | PCA | `07-Sl_ReduçãoDimensionalidade.pdf`, PDF pp. 3-12 | `src/models/dimensionality.py` | "I standardized the features and projected them onto two variance-maximizing components." |
| Section 31 | K-means | `03-Sl_AnaliseGrupos.pdf`, PDF pp. 8-18 | `src/models/clustering.py` | "K-means partitions standardized hourly profiles around three centroids." |
| Section 31 | DBSCAN | Same deck, PDF pp. 24-31 | `src/models/clustering.py` | "DBSCAN identifies density-connected groups and labels sparse observations as noise." |
| Section 31 | Silhouette score | Same deck, clustering evaluation topic, PDF pp. 49-52 | `evaluate_clustering_silhouette` | "The score summarizes within-cluster cohesion and between-cluster separation." |
| Sections 21 and 23 | Bootstrap confidence intervals | `05-Sl_AvaliacaoSelecaoModelos.pdf`, PDF pp. 11-17 | `bootstrap_metric_ci`, initial model reports | "I resampled holdout prediction pairs to estimate MAE uncertainty; this is not a block bootstrap." |
| Section 30 | Streamlit dashboard | Project-specific deployment extension | `src/dashboard/app.py` | "The dashboard turns generated outputs into an auditable presentation and monitoring interface." |

### Course-method snapshot disclosure

The committed initial feature-selection, outlier, PCA, clustering, and initial
model-comparison outputs were generated from an earlier **2,160-row hourly
snapshot**. The final hourly modelling datasets contain **17,544 timestamps**.
Present the 2,160-row results as an earlier course-method stage, not as results
computed on the final two-year snapshot. The final backtesting, multistep,
weather, and risk reports use the expanded data.

## 8. Validation strategy

### Why a random split is wrong

A random split can train on future observations and test on earlier ones. It
also breaks autocorrelation and seasonal order, producing an unrealistically
easy test. Time-series evaluation must train on the past and test on the future.

### Chronological holdout

The initial models and direct multi-step experiments use the earliest 80% for
training and the latest 20% for testing. Model-internal Ridge and LASSO tuning
uses `TimeSeriesSplit` rather than shuffled folds.

### Rolling-origin backtesting

- initial training window: 60% of usable observations;
- test window: 168 hours;
- step: 168 hours;
- folds: 41 for consumption and 41 for grid injection;
- training strategy: expanding chronological window.

The reported rolling-origin values are **fold means**, with accompanying fold
standard deviations in the backtesting summaries.

### Direct multi-step forecasting

Separate targets and models are created for 1h, 6h, 24h, and 168h. This avoids
recursive error accumulation and makes degradation by horizon explicit. The
complete multistep scenario grid uses one chronological holdout, not
rolling-origin evaluation.

### Robustness without lag 1

Removing lag 1 tests whether the model is genuinely useful when the most recent
observation is unavailable or delayed. Consumption MAE increases by roughly
35%-133%, depending on model. Grid injection MAE increases by roughly
363%-375%. This demonstrates strong nowcasting dependence, especially for grid
injection.

## 9. Results explanation

### Rolling-origin results

| Target | Best model by mean RMSE | Mean MAE | Mean RMSE | Mean MAPE | Mean R-squared | Folds |
|---|---|---:|---:|---:|---:|---:|
| Consumption | Random Forest | 21,507.19 | 32,405.79 | 5.26% | 0.973 | 41 |
| Grid injection | LASSO | 56,252.94 | 75,944.84 | 6.87% | 0.954 | 41 |

The consumption Random Forest won 39 of 41 folds. However, do not hide fold
variability: its RMSE standard deviation is 34,245.08 and its MAPE standard
deviation is 25.04 percentage points. The mean is strong, but some periods are
substantially harder.

Grid injection model dominance is less clear. LASSO and Random Forest each win
13 folds, Ridge wins 10, and Gradient Boosting wins 5. LASSO is selected because
it has the lowest mean RMSE, with Ridge extremely close.

### Direct multi-step results

| Target | Horizon | Best scenario/model | MAE | RMSE | MAPE | R-squared |
|---|---:|---|---:|---:|---:|---:|
| Consumption | 1h | Weather-enriched Random Forest | 34,623.88 | 51,690.22 | 2.29% | 0.965 |
| Consumption | 6h | Random Forest with lag 1 | 51,435.79 | 76,913.84 | 3.35% | 0.923 |
| Consumption | 24h | Random Forest with lag 1 | 57,737.44 | 85,133.36 | 3.75% | 0.905 |
| Consumption | 168h | Weather-enriched Random Forest | 71,841.30 | 99,941.94 | 4.63% | 0.870 |
| Grid injection | 1h | Weather-enriched LASSO | 101,409.10 | 133,984.68 | 11.89% | 0.938 |
| Grid injection | 6h | Gradient Boosting with lag 1 | 255,590.22 | 323,869.08 | 33.55% | 0.639 |
| Grid injection | 24h | Weather-enriched Ridge | 390,146.36 | 487,494.36 | 54.34% | 0.183 |
| Grid injection | 168h | Calendar/seasonal Ridge | 444,514.46 | 529,807.29 | 63.05% | 0.036 |

### Defensible interpretation

- Consumption is comparatively predictable across all tested horizons.
- Grid injection has a strong short-horizon result but degrades sharply.
- The 24h injection result is weak.
- The 168h injection result is weak and exploratory, with almost no explained
  variance.
- Do not present 24h or 168h injection as operationally established.

### Weather-enriched results

For consumption, historical weather improves the best 1h and 168h scenarios,
is effectively similar at 6h, and is worse at 24h. Weather is useful context but
does not replace lagged target history.

Operationally, historical reanalysis cannot be treated as a future weather
forecast. Deployment would require observations or forecasts genuinely
available at issue time, especially for longer horizons.

## 10. Operational risk proxy

```text
risk_score = 100 * (
    0.30 * pressure_score
  + 0.25 * seasonal_deviation_score
  + 0.20 * change_score
  + 0.15 * outlier_score
  + 0.10 * weather_score
)
```

### Components

- **Pressure score:** where the current target lies within a robust historical
  range.
- **Seasonal deviation score:** the largest deviation from daily, weekly, and
  rolling references.
- **Change score:** unusual short-term or daily relative change.
- **Outlier score:** whether IQR, z-score, or Isolation Forest identifies the
  observation as anomalous.
- **Weather score:** whether aligned weather is available and heavy rain or
  strong wind is flagged.

### Descriptive levels

| Score | Level |
|---:|---|
| `< 35` | Low |
| `>= 35 and < 60` | Medium |
| `>= 60 and < 80` | High |
| `>= 80` | Critical |

### Mandatory disclaimer

The weights and thresholds are heuristic and fixed by design. They were not
learned or calibrated against real failure labels. The score is not a
probability of failure and does not forecast confirmed outages. It is useful as
an interpretability and monitoring layer that ranks unusual or pressured
periods consistently.

## 11. Dashboard demo script

The dashboard is interactive but **not geographic**; no map is implemented.

1. **Executive overview**
   - Select `Consumption forecast` first.
   - Explain row count, missing targets, risk KPIs, and best observed RMSE.
   - State that forecast errors use E-REDES reported units.
2. **Time series**
   - Use daily mean for readability.
   - Explain the observed target on the left axis and risk proxy on the 0-100
     right axis.
   - Point out that high proxy values are not confirmed failures.
3. **Model comparison**
   - Compare horizons and scenarios.
   - Explain lower MAE/RMSE/MAPE and higher R-squared.
   - Switch to `Grid injection forecast` and show its stronger degradation.
4. **Risk events**
   - Show high and critical proxy-score events.
   - Explain component scores and weather/outlier filters.
   - Repeat: "Risk is a proxy, not a confirmed failure probability."
5. **Weather alignment**
   - Show complete 2024-2025 overlap.
   - Distinguish historical Open-Meteo reanalysis from IPMA
     operational/recent weather.
6. **Data quality**
   - Show coverage and missing targets.
   - Explain retained DST timestamp ambiguity and the separate consumption gap.
7. **Methodology disclaimer**
   - Summarize chronological validation, multi-horizon evaluation, and risk
     limitations.

## 12. Main limitations

- No labelled outage or failure target is available.
- The operational risk score is a heuristic proxy, not a calibrated failure
  probability.
- Historical Open-Meteo reanalysis is auxiliary and not an official IPMA
  historical station archive.
- Live E-REDES, IPMA, and Open-Meteo APIs are mutable; exact reproduction
  requires a matching snapshot and recorded hashes.
- Raw and 15-minute E-REDES layers contain 16 duplicate timestamps associated
  with DST transition dates. They contain different values and were not
  silently discarded.
- Missing quarter-hours total 104 for consumption and 8 each for injection and
  production. Consumption includes a separate 96-quarter-hour gap across
  2025-10-13/14.
- Hourly aggregation may bias the four DST-affected hours because repeated
  source records receive equal weight.
- Ordinary bootstrap MAE intervals resample observations independently and do
  not account for temporal dependence. A block bootstrap would be stronger.
- Grid injection forecasting is weak at 24h and exploratory at 168h.
- The multistep scenario grid uses one chronological holdout rather than
  rolling-origin validation.
- Weather-enriched experiments use forecast-origin reanalysis values rather
  than target-time operational forecasts.
- The final dashboard has no geographic map.
- The final modelling window covers two years and may not represent future
  structural changes.

## 13. Challenges and solutions

| Challenge | Problem | Solution | How to explain |
|---|---|---|---|
| Realistic objective | No direct labels supported failure prediction | Forecast measurable energy targets and use a transparent risk proxy | "I changed the claim to match the available evidence." |
| Dataset choice | E-REDES, IPMA, Open-Meteo, and REN offered overlapping possibilities | Use E-REDES targets, IPMA context, historical Open-Meteo enrichment, and defer REN | "I prioritized target relevance and a feasible end-to-end scope." |
| API extraction | E-REDES required pagination and bounded extraction | Add date chunks, page limits, ordering, retries, and local snapshots | "The extraction is controlled and auditable rather than an unbounded download." |
| Weather alignment | Recent IPMA data did not overlap 2024-2025 | Do not force-fill; add historical Open-Meteo reanalysis | "I solved the temporal mismatch without inventing historical IPMA values." |
| Temporal granularity | Energy was 15-minute while weather was hourly | Preserve 15-minute data and create an hourly modelling layer | "Hourly data aligns sources and makes daily/weekly seasonality interpretable." |
| Leakage prevention | Lags, rolling windows, and future targets can leak information | Sort chronologically, use backward shifts, shift before rolling, and validate feature alignment | "Every target-derived feature uses information from before the prediction target." |
| Model validation | One holdout can depend on one unusual future period | Add 41-fold expanding rolling-origin backtesting | "I tested the models repeatedly on future weekly blocks." |
| Lag-1 dependence | Strong results relied heavily on the most recent observation | Remove lag 1 in robustness tests and add direct multi-step forecasting | "I separated excellent nowcasting from realistic longer-horizon performance." |
| Dashboard communication | CSV metrics and plots were difficult to present coherently | Build target-specific tabs, explanations, risk warnings, and data-quality views | "The dashboard is an interpretation layer, not a new model." |

## 14. Future work

1. Integrate labelled outage, interruption, or service-quality events.
2. Add REN Data Hub as broader national energy-system context.
3. Build regional models if regional consumption and injection targets become
   available.
4. Replace origin-time reanalysis with weather observations or forecasts
   genuinely available for each target horizon.
5. Run rolling-origin validation for a reduced set of winning multistep models.
6. Add a map or geographic layer when region-level targets and coordinates are
   available.
7. Train and calibrate a supervised risk model only after obtaining trustworthy
   labelled events.
8. Use block bootstrap or another dependence-aware uncertainty method.
9. Investigate generation and renewable forecasts to improve long-horizon grid
   injection.

## 15. Final conclusion

"This project demonstrates a complete and defensible data science workflow for
Portugal's electricity distribution context. It combines controlled data
engineering, leakage-aware feature creation, multiple course techniques,
chronological validation, direct multi-horizon forecasting, and an interpretable
monitoring proxy. The strongest result is short-horizon consumption forecasting.
Grid injection is useful at short horizons but remains weak at longer horizons.
The project does not claim to predict failures; it shows how far the available
public data can support forecasting and transparent operational interpretation."

## 16. What not to say

Avoid these statements:

- "We predict electricity grid failures."
- "The risk score is a probability of failure."
- "The high and critical bands are calibrated from outage events."
- "IPMA provides the historical 2024-2025 training weather."
- "Open-Meteo is official IPMA data."
- "REN was integrated into the pipeline."
- "The dashboard is geographic" or "the dashboard contains a map."
- "The 168h grid injection model is strong."
- "All reported course-method results use the final 17,544-hour dataset."
- "The rolling-origin numbers are one aggregate test result."
- "Weather always improves the models."
- "The project is exactly reproducible from live APIs at any future date."
- "`validate_pipeline` fails automatically on every data-quality issue."
- "Random Forest is always the best model."

Prefer these alternatives:

- "The project forecasts measurable consumption and grid injection targets."
- "The operational risk proxy ranks unusual conditions using heuristic,
  explainable components."
- "Historical Open-Meteo reanalysis fills the 2024-2025 weather-coverage gap."
- "REN remains a proposed future extension."
- "The dashboard is an interactive analytical presentation layer."
- "The 168h injection result is weak and exploratory."

## 17. Likely professor questions

### Why did you not use a random train/test split?

Random splitting would mix future and past observations, break temporal order,
and overestimate performance. I trained on earlier timestamps and tested only
on later timestamps.

### Why add rolling-origin backtesting after a chronological holdout?

A single holdout represents only one future period. Rolling-origin backtesting
repeats the forecasting exercise across 41 future weekly blocks and reveals
variation over time.

### Are the reported backtesting metrics averages?

Yes. The headline MAE, RMSE, MAPE, and R-squared values are means across 41
folds. The backtesting summaries also report standard deviations.

### Why is consumption easier to forecast than injection?

Consumption has stronger recurring temporal behavior. Grid injection depends
more heavily on volatile renewable generation and external production
conditions that are not fully represented by target-history features.

### Is the 168h injection model useful?

Only as an exploratory baseline. Its best R-squared is approximately 0.036 and
MAPE is approximately 63%, so I do not claim established operational value.

### Why did removing lag 1 hurt injection so much?

Injection is highly persistent at short horizons. Lag 1 contains the latest
state of that volatile process. Removing it exposes the limits of the remaining
calendar, seasonal, and older-lag information.

### Is the risk score predictive?

It is contemporaneous and descriptive. It combines observed pressure,
deviation, change, outlier, and weather indicators. It is not trained against
future failures.

### How were the risk weights selected?

They are fixed heuristic weights chosen for transparency and to balance five
interpretable components. They are not statistically calibrated. Calibration
would require labelled events.

### Why use Open-Meteo if IPMA is the official source?

The IPMA endpoints used provide operational/recent context but not complete
hourly 2024-2025 coverage. Historical Open-Meteo reanalysis solves that specific
alignment problem without pretending that recent IPMA observations are
historical.

### Did weather improve forecasting?

At selected horizons. It improved consumption at 1h and 168h, was similar at
6h, and worsened the best 24h comparison. Lag history remained more important.

### Could historical reanalysis be used directly in production?

Not as future information. A deployed model would need weather observations or
forecasts available at the issue time. The experiment demonstrates historical
association and incremental forecasting value, not a complete operational
weather service.

### Why was REN excluded?

E-REDES already supplied the required prediction targets. REN would add useful
system context but also another extraction and alignment problem. I prioritized
a complete, validated pipeline and retained REN as future work.

### What exactly happened with DST timestamps?

The source snapshot contains repeated UTC timestamps with different values on
Portuguese DST transition dates. I did not invent offsets or silently discard
measurements. The ambiguity is documented, and complete hourly chronology is
retained.

### Why are the course-method outputs based on only 2,160 rows?

They were produced during an earlier hourly implementation stage. I retain them
as evidence of the course-method workflow and disclose their snapshot size. The
final rolling-origin, multistep, weather, and risk analyses use the expanded
17,544-hour datasets.

### Did feature selection determine the final model inputs automatically?

No. Correlation and Random Forest importance were implemented as course-method
analyses. Final forecast features were deliberately constrained to
leakage-safe calendar, lag, and rolling variables.

### Are the bootstrap confidence intervals time-series aware?

No. They use ordinary independent resampling of holdout prediction pairs. They
are useful as an initial uncertainty estimate, but block bootstrap would better
respect serial dependence.

### Does the dashboard contain a geographic map?

No. It is an interactive analytical dashboard with time series, forecast
comparisons, risk events, weather, data quality, and methodology. Geographic
mapping remains future work.

### Does `validate_pipeline` guarantee all data are valid?

No. It is an inspection-oriented command that reports schemas, coverage,
duplicates, and missing values. Other validators contain stronger assertions,
and the timestamp audit documents known data-quality limitations.

### What is the strongest contribution of the project?

The strongest contribution is not one algorithm. It is the defensible workflow:
controlled extraction, layered data preparation, leakage-aware features,
chronological evaluation, honest horizon-specific interpretation, and explicit
separation of forecasting from proxy risk scoring.
