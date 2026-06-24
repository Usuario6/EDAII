# Direct Multi-Step Forecasting Report

## Evaluation Scope

Direct models predict each horizon independently. This tests whether the nowcasting results remain useful when the latest lag is unavailable, without recursively feeding predictions back as observations.

Horizons: 1, 6, 24, and 168 hours. Models: seasonal naive, Ridge, LASSO, Random Forest, and Gradient Boosting. Evaluation uses a chronological 80/20 holdout; rolling-origin evaluation is deferred because the full scenario grid would be computationally expensive.

## Forecasting Results by Prediction Horizon

```text
    dataset  horizon          scenario             model        mae       rmse   mape    r2
consumption        1  weather_enriched     random_forest  34623.878  51690.224  2.295 0.965
consumption        6         with_lag1     random_forest  51435.786  76913.843  3.348 0.923
consumption       24         with_lag1     random_forest  57737.438  85133.364  3.747 0.905
consumption      168  weather_enriched     random_forest  71841.296  99941.940  4.634 0.870
  injection        1  weather_enriched             lasso 101409.099 133984.679 11.895 0.938
  injection        6         with_lag1 gradient_boosting 255590.222 323869.077 33.554 0.639
  injection       24  weather_enriched             ridge 390146.358 487494.360 54.341 0.183
  injection      168 calendar_seasonal             ridge 444514.461 529807.291 63.046 0.036
```

## Impact of Recent Observations (Lag 1)

```text
    dataset  horizon  mae_change_pct  rmse_change_pct
consumption        1           54.26            54.82
consumption        6            5.56             6.52
consumption       24            9.54             8.45
consumption      168            7.74             7.23
  injection        1          169.36           158.70
  injection        6           31.35            29.51
  injection       24            7.75             6.02
  injection      168           -0.02            -0.04
```

### Consumption degradation by horizon

```text
horizon
1       0.00
6      48.80
24     64.70
168    93.35
```

### Injection degradation by horizon

```text
horizon
1        0.00
6      141.72
24     263.84
168    295.42
```

## Interpretation of Results

Historically aligned weather features were usable for this experiment. No weather values were force-filled across the 2024–2025 interval.
Use the strongest with-lag model for next-hour nowcasting. For operational horizons where lag 1 is unavailable, select from the without-lag or calendar/seasonal scenarios and treat the performance loss as the realistic forecast cost.
Injection forecasts are credible primarily at short horizons. The 24-hour result has weak explanatory power, and the 168-hour result is weak and exploratory; neither should be presented as established operational performance.

## Known Limitations

The experiment uses one chronological holdout and a bounded two-year energy window. Weather features describe forecast-origin conditions rather than target-time forecasts, and calendar features include national holidays without local detail.
