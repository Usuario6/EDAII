# Direct multi-step forecasting report

## Purpose

Direct models predict each horizon independently. This tests whether the nowcasting results remain useful when the latest lag is unavailable, without recursively feeding predictions back as observations.

Horizons: 1, 6, 24, and 168 hours. Models: seasonal naive, Ridge, LASSO, Random Forest, and Gradient Boosting. Evaluation uses a chronological 80/20 holdout; rolling-origin evaluation is deferred because the full scenario grid would be computationally expensive.

## Best result per horizon

```text
    dataset  horizon          scenario             model        mae       rmse   mape    r2
consumption        1         with_lag1     random_forest  35650.532  53134.939  2.351 0.963
consumption        6         with_lag1     random_forest  51435.786  76913.843  3.348 0.923
consumption       24         with_lag1     random_forest  57737.438  85133.364  3.747 0.905
consumption      168 calendar_seasonal     random_forest  75541.790 100974.608  4.872 0.867
  injection        1         with_lag1             lasso 101310.530 134381.599 11.745 0.938
  injection        6         with_lag1 gradient_boosting 255590.222 323869.077 33.554 0.639
  injection       24         with_lag1             ridge 395163.486 491312.828 54.744 0.170
  injection      168 calendar_seasonal             ridge 444514.461 529807.291 63.046 0.036
```

## Lag-1 dependence

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
6      44.75
24     60.22
168    90.03
```

### Injection degradation by horizon

```text
horizon
1        0.00
6      141.01
24     265.61
168    294.26
```

## Weather and recommendation

Historically aligned weather features were not usable for this experiment. No weather values were force-filled across the 2024–2025 interval.
Use the strongest with-lag model for next-hour nowcasting. For operational horizons where lag 1 is unavailable, select from the without-lag or calendar/seasonal scenarios and treat the performance loss as the realistic forecast cost.
Limitations: one chronological holdout, a bounded two-year energy window, no historically overlapping weather, and no holiday-locality features beyond national Portuguese holidays.
