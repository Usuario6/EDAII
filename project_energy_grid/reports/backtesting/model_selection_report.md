# Rolling-origin model selection report

## Consumption

- Coverage: 2024-01-01 00:00:00+00:00 to 2025-12-31 23:00:00+00:00
- Hourly rows: 17,544
- Models: random_forest, gradient_boosting, ridge, lasso, seasonal_naive_lag_24
- Rolling origins: 41 folds; initial usable train rows 10,525; test window 168 hours; weekly step.
- Recommended candidate by average RMSE: **random_forest**.

Average and standard deviation by model:

```text
                model   mae_mean   mae_std  rmse_mean  rmse_std  mape_mean  mape_std  r2_mean  r2_std  folds
        random_forest  21507.194 11555.961  32405.788 34245.076      5.261    25.039    0.973   0.074     41
    gradient_boosting  31783.113 14527.022  44810.941 37449.429      6.379    27.849    0.955   0.091     41
                ridge  49510.270 13558.135  63449.992 24373.924      5.774    16.082    0.920   0.051     41
                lasso  49513.515 13532.431  63463.533 24360.544      5.778    16.107    0.920   0.051     41
seasonal_naive_lag_24 107975.908 34608.568 159490.239 58340.670     11.420    27.634    0.501   0.266     41
```

## Injection

- Coverage: 2024-01-01 00:00:00+00:00 to 2025-12-31 23:00:00+00:00
- Hourly rows: 17,544
- Models: lasso, ridge, gradient_boosting, random_forest, seasonal_naive_lag_24
- Rolling origins: 41 folds; initial usable train rows 10,525; test window 168 hours; weekly step.
- Recommended candidate by average RMSE: **lasso**.

Average and standard deviation by model:

```text
                model   mae_mean    mae_std  rmse_mean   rmse_std  mape_mean  mape_std  r2_mean  r2_std  folds
                lasso  56252.939  11475.153  75944.838  18367.761      6.873     2.765    0.954   0.029     41
                ridge  56284.474  11418.772  75953.029  18342.239      6.867     2.683    0.954   0.029     41
    gradient_boosting  57986.521  12641.280  77379.319  19585.791      7.794     6.903    0.952   0.030     41
        random_forest  57796.220  12639.061  77840.143  19101.173      7.451     5.565    0.951   0.031     41
seasonal_naive_lag_24 389346.704 167299.478 493504.633 208970.071     49.325    21.195   -0.475   0.601     41
```

## Interpretation and limitations

Lag-1 dominance is reported explicitly: strong degradation without it indicates that short-term persistence drives much of the score.
The evaluation uses an expanding training window and non-overlapping chronological test windows; no random split is used.
The current weather observations do not overlap the E-REDES history, so weather is not included.
Results cover a bounded historical interval and do not prove performance during unseen structural changes.
