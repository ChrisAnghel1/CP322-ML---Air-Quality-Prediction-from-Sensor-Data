## Model training and evaluation

These scripts fit and compare the models that predict `CO(GT)` from the eight sensor and weather columns in `air_quality_clean.csv`. All use the same split, so their results can be compared directly.

| Script | Purpose | Output files (in `results/`) |
| --- | --- | --- |
| `src/train_linear.py` | Linear regression and mean baseline | `linear_metrics.csv`, `linear_coefficients.csv`, `linear_pred_vs_actual.png`, `linear_residuals.png` |
| `src/train_ridge.py` | Ridge regression | `ridge_metrics.csv`, `ridge_coefficients.csv`, `ridge_alpha_curve.png`, `ridge_coefficient_path.png` |
| `src/train_random_forest.py` | Random forest | `rf_grid_search.csv`, `rf_metrics.csv`, `rf_feature_importance.csv`, `rf_pred_vs_actual.png`, `rf_residuals.png` |
| `src/evaluate_test.py` | Final comparison on the test set | `test_metrics.csv`, `test_error_by_co_level.csv`, `test_permutation_importance.csv`, `test_model_comparison.png`, `test_pred_vs_actual.png`, `test_timeline.png`, `test_permutation_importance.png` |

Run them from the repo root, for example `python src/train_linear.py`. Each takes an optional path to a different CSV as its first argument.

### Shared setup

**Features and target.** The target is `CO(GT)`. The features are the five `PT08` sensor columns plus `T`, `RH` and `AH`.

**Chronological split.** Rows are cut by position, with no shuffling, because the data is an hourly time series and a random split would leak information between neighbouring hours.

| Set | Share | Rows | Dates |
| --- | --- | --- | --- |
| Train | first 70% | 5,140 | 10 Mar 2004 – 19 Dec 2004 |
| Validation | next 15% | 1,102 | 19 Dec 2004 – 16 Feb 2005 |
| Test | last 15% | 1,102 | 16 Feb 2005 – 4 Apr 2005 |

The three training scripts use only the train and validation sets. The test set is scored only by `evaluate_test.py`.

**Scaling.** For linear and Ridge regression, features are standardized (mean 0, standard deviation 1). The scaler is fitted on the training rows only and then applied to the later sets, so no information from them reaches the model. This makes coefficients comparable across features, and Ridge needs it because its penalty treats all coefficients equally. The random forest uses the unscaled features, since tree splits do not depend on feature scale.

**Metrics.** RMSE and MAE are in the units of `CO(GT)` (mg/m³), and R² is the share of variance explained. A baseline that predicts the training mean for every row is included for reference.

### Linear regression (`train_linear.py`)

The script fits scikit-learn's `LinearRegression` on the scaled training data, scores it on train and validation, prints the standardized coefficients and the feature correlation matrix, and saves two diagnostic plots.

**Results**

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Linear regression, train | 0.430 | 0.297 | 0.912 |
| Linear regression, validation | 0.686 | 0.452 | 0.794 |
| Mean baseline, validation | 1.514 | 1.183 | -0.003 |

Validation RMSE is about 55% lower than the baseline. The drop in R² from train to validation is expected to some extent with a time-ordered split, since validation comes from a later, winter, part of the year.

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
- The residual plot shows errors that come in episodes (for example, large underprediction in late December and overprediction around 22 to 24 January), not a steady drift. The predicted-vs-actual plot shows the model underpredicting above a CO value of about 5. The straight lines in the residual plot are gaps where rows were dropped.

### Ridge regression (`train_ridge.py`)

Ridge is linear regression with a penalty on the size of the coefficients, which helps when features are correlated. The script tries 61 penalty strengths (`alpha`) from 0.001 to 1,000, spaced evenly on a log scale. For each one it fits a model on the training data and records the validation RMSE. It keeps the alpha with the lowest validation RMSE, refits at that value, and compares the result with plain linear regression.

**Results**
- The best alpha is 39.81.
- Validation RMSE is flat at about 0.686 for alpha below roughly 1, dips slightly to about 0.685 at the best alpha, and rises to about 0.714 at alpha = 1,000. The improvement over linear regression is about 0.001, which is negligible.

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Ridge, train | 0.432 | 0.299 | 0.911 |
| Ridge, validation | 0.685 | 0.453 | 0.795 |

**Coefficient behaviour** (from `ridge_coefficient_path.png`)
- `PT08.S2(NMHC)` stays dominant at every reasonable alpha, shrinking from about 1.40 to 1.24 at the best alpha. `PT08.S1(CO)` barely changes.
- `PT08.S4(NO2)` and `PT08.S5(O3)` shrink toward zero (-0.083 and -0.045 at the best alpha) and turn positive under heavier penalties, which suggests their signs in the unpenalized model are not reliable.

**Observations**
- Ridge gives essentially the same accuracy as linear regression, so the correlation among the sensors does not hurt prediction much. It mainly affects how far the coefficients of the weaker sensors can be trusted.

### Random forest (`train_random_forest.py`)

A random forest averages many decision trees, so it can capture nonlinear relationships that the linear models cannot. The script tries 20 combinations of `max_depth` (5, 10, 15, unlimited) and `min_samples_leaf` (1, 5, 20, 50, 100), each with 200 trees and `random_state=0`. It fits each on the training data, keeps the combination with the lowest validation RMSE, and refits it.

**Grid search** (validation RMSE)

| `max_depth` | leaf = 1 | leaf = 5 | leaf = 20 | leaf = 50 | leaf = 100 |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.704 | 0.699 | 0.695 | 0.692 | 0.697 |
| 10 | 0.751 | 0.708 | 0.687 | **0.685** | 0.695 |
| 15 | 0.754 | 0.708 | 0.687 | 0.685 | 0.695 |
| unlimited | 0.754 | 0.708 | 0.687 | 0.685 | 0.695 |

The selected settings are `max_depth=10`, `min_samples_leaf=50`. At depths of 10 and above, very small leaves give the worst scores, which is consistent with overfitting. With 20 or more samples per leaf, those three depths score the same to three decimals, so the choice of depth among them makes no practical difference.

**Results**

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Random forest, train | 0.374 | 0.241 | 0.933 |
| Random forest, validation | 0.685 | 0.442 | 0.795 |

**Feature importances** (impurity-based, from the training data)

| Feature | Importance |
| --- | --- |
| `PT08.S2(NMHC)` | 0.959 |
| `PT08.S1(CO)` | 0.022 |
| `T` | 0.017 |
| `PT08.S4(NO2)` | 0.001 |
| `PT08.S5(O3)` | 0.001 |
| `AH` | 0.000 |
| `RH` | 0.000 |
| `PT08.S3(NOx)` | 0.000 |

**Observations**
- On validation, the random forest has the same RMSE as Ridge to three decimals and a slightly lower MAE (0.442 against 0.453).
- The gap between train and validation RMSE is larger than for the linear models.
- The forest relies almost entirely on `PT08.S2(NMHC)`.

### Final test evaluation (`evaluate_test.py`)

The script repeats the Ridge and random forest searches on the validation set, which selects the same settings as above. It then fits each model on the training rows only (not training plus validation) and scores it on validation and test.

**Results**

| Model | Validation RMSE | Test RMSE | Test MAE | Test R² |
| --- | --- | --- | --- | --- |
| Mean baseline | 1.514 | 1.306 | 1.046 | -0.028 |
| Linear regression | 0.686 | 0.503 | 0.359 | 0.847 |
| Ridge | 0.685 | 0.501 | 0.355 | 0.849 |
| Random forest | 0.685 | 0.483 | 0.330 | 0.859 |

**Error by CO level** (test set). Mean bias is predicted minus actual, so a negative value means underprediction.

| CO band (mg/m³) | Rows | Linear MAE | Ridge MAE | Forest MAE | Linear bias | Ridge bias | Forest bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 to under 1 | 258 | 0.355 | 0.352 | 0.248 | -0.067 | -0.043 | 0.185 |
| 1 to under 2 | 403 | 0.262 | 0.260 | 0.222 | -0.100 | -0.070 | -0.076 |
| 2 to under 4 | 350 | 0.344 | 0.340 | 0.399 | -0.145 | -0.118 | -0.252 |
| 4 and above | 91 | 0.854 | 0.839 | 0.767 | -0.799 | -0.783 | -0.656 |

**Permutation importance** (increase in test RMSE when a feature is shuffled, averaged over 20 repeats). It is computed for linear regression and the random forest only.

| Feature | Linear regression | Random forest |
| --- | --- | --- |
| `PT08.S2(NMHC)` | 1.347 | 1.175 |
| `PT08.S1(CO)` | 0.174 | 0.027 |
| `T` | 0.035 | 0.021 |
| `PT08.S4(NO2)` | 0.035 | 0.003 |
| `PT08.S5(O3)` | 0.025 | 0.002 |
| `PT08.S3(NOx)` | 0.067 | 0.000 |
| `AH` | -0.001 | 0.000 |
| `RH` | 0.001 | 0.000 |

**Observations**
- The random forest has the lowest test RMSE, but only by about 0.02. Since the models were nearly tied on validation, this is weak evidence that it is better in general.
- The forest has lower MAE than the linear models below 2 mg/m³ and at 4 mg/m³ and above, but higher MAE from 2 to 4 mg/m³.
- All three models underpredict most at 4 mg/m³ and above.
- Every model, including the baseline, has lower RMSE on test than on validation. This suggests the test period is easier to predict, although the cause has not been investigated.
- Both importance measures put `PT08.S2(NMHC)` far ahead of every other feature.

### Limitations

- Linear and Ridge regression cannot capture nonlinear sensor responses. The random forest can, but it still underpredicts at high CO values.
- Validation and test cover only late December to early April, so the scores may not represent the whole year.
- Coefficients and importances for correlated sensors should not be interpreted individually.
- Ridge and random forest settings are chosen on the validation set, so their validation scores are slightly optimistic. The test set is the fairer comparison.
