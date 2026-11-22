## Linear regression and Ridge models

These two scripts fit the baseline models for predicting `CO(GT)` from the eight sensor and weather columns in `air_quality_clean.csv`. Both use the same split and scaling, so their results can be compared directly.

| Script | Model | Output files (in `results/`) |
| --- | --- | --- |
| `src/train_linear.py` | Linear regression | `linear_metrics.csv`, `linear_coefficients.csv`, `linear_pred_vs_actual.png`, `linear_residuals.png` |
| `src/train_ridge.py` | Ridge regression | `ridge_metrics.csv`, `ridge_coefficients.csv`, `ridge_alpha_curve.png`, `ridge_coefficient_path.png` |

Run both from the repo root, for example `python src/train_linear.py`. Each takes an optional path to a different CSV as its first argument.

### Shared setup

**Features and target.** The target is `CO(GT)`. The features are the five `PT08` sensor columns plus `T`, `RH` and `AH`.

**Chronological split.** Rows are cut by position, with no shuffling, because the data is an hourly time series and a random split would leak information between neighbouring hours.

| Set | Share | Rows |
| --- | --- | --- |
| Train | first 70% | 5,140 |
| Validation | next 15% | 1,102 |
| Test | last 15% | 1,102 |

The test set is not used in either script. It is kept for the final comparison once all models are finished. The validation window covers roughly 20 December 2004 to 17 February 2005, so it is mostly winter.

**Scaling.** Features are standardized (mean 0, standard deviation 1). The scaler is fitted on the training rows only and then applied to validation, so no validation information reaches the model. This makes coefficients comparable across features, and Ridge needs it because its penalty treats all coefficients equally.

**Metrics.** RMSE and MAE are in the units of `CO(GT)` (mg/m³), and R² is the share of variance explained. A baseline that predicts the training mean for every row is included for reference.

### Linear regression (`train_linear.py`)

The script fits scikit-learn's `LinearRegression` on the scaled training data, scores it on train and validation, prints the standardized coefficients and the feature correlation matrix, and saves two diagnostic plots.

**Results**

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Linear regression, train | 0.430 | 0.297 | 0.912 |
| Linear regression, validation | 0.686 | 0.452 | 0.794 |
| Mean baseline, validation | 1.514 | 1.183 | -0.003 |

Validation error is about 55% lower than the baseline. The drop in R² from train to validation is expected to some extent with a time-ordered split, since validation comes from a later, mostly winter, part of the year.

**Standardized coefficients**

| Feature | Coefficient |
| --- | --- |
| `PT08.S2(NMHC)` | 1.397 |
| `PT08.S1(CO)` | 0.393 |
| `PT08.S3(NOx)` | 0.200 |
| `PT08.S4(NO2)` | -0.145 |
| `T` | -0.141 |
| `PT08.S5(O3)` | -0.111 |
| `AH` | 0.044 |
| `RH` | 0.036 |

**Observations**
- `PT08.S2(NMHC)` is by far the strongest predictor, even though it is the hydrocarbon sensor rather than the CO sensor.
- Several sensors are highly correlated with each other (`PT08.S1`, `S2` and `S5` are at about 0.89 to 0.90), so individual coefficients for the weaker sensors may be unstable. The negative signs on `PT08.S4(NO2)` and `PT08.S5(O3)` are probably an artifact of this, but this has not been tested directly.
- The residual plot shows errors that come in episodes (for example, large underprediction in late December and overprediction around 22 to 24 January), not a steady drift. The predicted-vs-actual plot shows the model underpredicting above a CO value of about 5. The straight lines in the residual plot are gaps where rows were dropped for a missing target.

### Ridge regression (`train_ridge.py`)

Ridge is linear regression with a penalty on the size of the coefficients, which helps when features are correlated. The script tries 61 penalty strengths (`alpha`) from 0.001 to 1,000, spaced evenly on a log scale. For each one it fits a model on the training data and records the validation RMSE. It keeps the alpha with the lowest validation RMSE, refits at that value, and compares the result with plain linear regression.

**Results**
- The best alpha is about 40, read from the alpha-curve plot. The exact value is printed when the script runs.
- Validation RMSE is flat at about 0.686 for alpha below roughly 1, dips slightly to about 0.685 near alpha = 40, and rises to about 0.714 at alpha = 1,000. The improvement over linear regression is about 0.001, which is negligible.

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Ridge, train | 0.432 | 0.299 | 0.911 |
| Ridge, validation | 0.685 | 0.453 | 0.795 |

**Coefficient behaviour** (from `ridge_coefficient_path.png`)
- `PT08.S2(NMHC)` stays dominant at every reasonable alpha, shrinking from about 1.4 to about 1.25 at the best alpha. `PT08.S1(CO)` barely changes.
- `PT08.S4(NO2)` and `PT08.S5(O3)` shrink to near zero at the best alpha and turn positive under heavier penalties, which suggests their signs in the unpenalized model are not reliable.

**Observations**
- Ridge gives essentially the same accuracy as linear regression, so the correlation among the sensors does not hurt prediction much. It mainly affects how far the coefficients of the weaker sensors can be trusted.
- Alpha is chosen on the validation set, so the Ridge validation score is slightly optimistic. The test set is the fair final comparison.

### Limitations of these two models

- Both are linear, so they cannot capture nonlinear sensor responses, and they underpredict at high CO values.
- Validation covers only about two months of winter, so the scores may not represent the whole year.
- Coefficients for correlated sensors should not be interpreted individually.
- The test set has not been used yet.