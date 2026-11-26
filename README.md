# CP322 Final Group Project — Air Quality Prediction

Regression project for CP322: predicting ground-truth pollutant concentration from
low-cost metal-oxide sensor readings, using the UCI Air Quality dataset.

## Problem

Given hourly readings from an array of five metal-oxide chemical sensors plus
temperature and humidity, predict the carbon monoxide concentration, `CO(GT)` in
mg/m³, measured by a co-located reference analyzer. Task type: regression.

## Results

Test set scores (the last 15% of rows, 16 February to 4 April 2005):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Mean baseline | 1.306 | 1.046 | -0.028 |
| Linear regression | 0.503 | 0.359 | 0.847 |
| Ridge | 0.501 | 0.355 | 0.849 |
| **Random forest** | **0.483** | **0.330** | **0.859** |

Validation set scores (19 December 2004 to 16 February 2005), used to choose
model settings:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Mean baseline | 1.514 | 1.183 | -0.003 |
| Linear regression | 0.686 | 0.452 | 0.794 |
| Ridge (`alpha` ≈ 39.8) | 0.685 | 0.453 | 0.795 |
| Random forest (`max_depth=10`, `min_samples_leaf=50`) | 0.685 | 0.442 | 0.795 |

- All three models cut test RMSE by more than 60% compared with the mean baseline.
- The random forest scores best on the test set, but its margin is small (RMSE
  about 0.02 below linear regression). On validation, all three models have nearly
  identical RMSE.
- `PT08.S2(NMHC)`, the hydrocarbon sensor, is by far the most important feature for
  both linear regression and the random forest, ahead of the CO sensor.
- All models underpredict high concentrations. For the 91 test rows at 4 mg/m³ or
  above, the mean bias is between -0.66 (random forest) and -0.80 (linear
  regression).
- The random forest is not best everywhere: in the 2–4 mg/m³ band its MAE is 0.399,
  against 0.340–0.344 for the linear models.

## Dataset

- Source: [UCI Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality)
- 9,357 hourly records, March 2004 – April 2005, from a polluted area in an Italian city
- 15 columns:
  - `Date`, `Time`
  - five reference-analyzer concentrations: `CO(GT)`, `NMHC(GT)`, `C6H6(GT)`,
    `NOx(GT)`, `NO2(GT)`
  - five metal-oxide sensor responses: `PT08.S1(CO)`, `PT08.S2(NMHC)`,
    `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`
  - `T` (temperature), `RH` (relative humidity), `AH` (absolute humidity)
- The repo uses `AirQualityUCI.xlsx`. UCI also provides a CSV version, which uses
  comma decimals and has trailing empty columns.
- Missing readings are coded as the literal value `-200`, not a blank cell, so this
  must be replaced with `NaN` before any analysis.
- `NMHC(GT)` is missing in about 90% of rows. It is dropped as unusable rather than
  imputed.
- In 366 rows, all five `PT08` sensors are missing at once, along with `C6H6(GT)`,
  `T`, `RH` and `AH`. These are sensor-array outages rather than single-channel gaps.
- The target, `CO(GT)`, is missing in about 18% of the raw rows.

## Preprocessing

`src/preprocessing.py` goes from 9,357 raw rows to 7,344 clean rows:

1. Combine `Date` and `Time` into a sorted `datetime` index.
2. Replace `-200` with `NaN`.
3. Drop the `NMHC(GT)` column.
4. Drop the 366 sensor-outage rows (all `PT08` columns missing): 8,991 rows left.
5. Drop rows with a missing `CO(GT)`: 7,344 rows left.
6. Drop `NOx(GT)`, `NO2(GT)` and `C6H6(GT)` so that reference measurements are not
   used as features.

The output, `air_quality_clean.csv`, has the `datetime` index, the `CO(GT)` target
and 8 features (five `PT08` sensors, `T`, `RH`, `AH`), with no missing values. See
[docs/data_pipeline.md](docs/data_pipeline.md) for details.

## Methods and experimental setup

**Split.** Rows are split in time order with no shuffling, because neighbouring
hours are related and a random split would leak information between sets.

| Set | Share | Rows | Dates |
|---|---|---|---|
| Train | first 70% | 5,140 | 10 Mar 2004 – 19 Dec 2004 |
| Validation | next 15% | 1,102 | 19 Dec 2004 – 16 Feb 2005 |
| Test | last 15% | 1,102 | 16 Feb 2005 – 4 Apr 2005 |

**Models.**

| Model | Input features | Settings tried on validation |
|---|---|---|
| Mean baseline | none (predicts the training mean) | none |
| Linear regression | standardized | none |
| Ridge | standardized | 61 values of `alpha`, log-spaced from 0.001 to 1,000 |
| Random forest | unscaled | `max_depth` in {5, 10, 15, None} × `min_samples_leaf` in {1, 5, 20, 50, 100}; 200 trees, `random_state=0` |

The scaler is fitted on the training rows only. Settings are chosen by lowest
validation RMSE. For the final comparison, each model is fitted on the training rows
only (not training plus validation) and scored on the test set.

**Metrics.** RMSE and MAE (both in mg/m³) and R². The final evaluation also reports
error and bias by CO level (0–1, 1–2, 2–4 and 4+ mg/m³) and permutation feature
importance on the test set for linear regression and the random forest.

More detail on each model and the final evaluation is in
[docs/training_models.md](docs/training_models.md).

## Limitations

- The data covers about 13 months from one location, so the results may not carry
  over to other sites or sensor units.
- Validation and test together cover only late December to early April, so neither
  measures performance in summer or autumn.
- Several sensors are highly correlated (`PT08.S1`, `S2` and `S5` at about
  0.89–0.90 on the training set), so individual coefficients and feature importances
  should be interpreted with care.
- Validation scores for Ridge and the random forest are slightly optimistic because
  their settings were chosen on that set.

## Repo structure

```
.
├── AirQualityUCI.xlsx           # raw dataset
├── air_quality_clean.csv        # output of preprocessing.py
├── docs/
│   ├── data_pipeline.md         # data loading and cleaning
│   └── training_models.md       # model training and evaluation
├── results/                     # metrics (CSV) and figures (PNG)
├── src/
│   ├── dataset.py               # exploration: missing %, outage rows, usable rows per target
│   ├── preprocessing.py         # cleaning, writes air_quality_clean.csv
│   ├── train_linear.py          # linear regression and mean baseline
│   ├── train_ridge.py           # Ridge with alpha search
│   ├── train_random_forest.py   # random forest with grid search
│   └── evaluate_test.py         # final comparison on the test set
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 or earlier: `requirements.txt` pins NumPy below 2.0, and those
releases do not support newer Python versions. The scripts were checked with
Python 3.12.

```bash
git clone https://github.com/ChrisAnghel1/CP322-ML---Air-Quality-Prediction-from-Sensor-Data.git
cd CP322-ML---Air-Quality-Prediction-from-Sensor-Data
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all scripts from the repo root:

```bash
python src/dataset.py               # print missing-data stats
python src/preprocessing.py         # write air_quality_clean.csv
python src/train_linear.py          # linear regression
python src/train_ridge.py           # Ridge regression
python src/train_random_forest.py   # random forest
python src/evaluate_test.py         # test-set comparison
```

`preprocessing.py` accepts an optional input `.xlsx` path as its first argument. The
training and evaluation scripts accept an optional path to a cleaned CSV.
`evaluate_test.py` repeats the hyperparameter searches itself, so it only needs
`air_quality_clean.csv`.

### Outputs

All are written to `results/`.

| Script | Files |
|---|---|
| `train_linear.py` | `linear_metrics.csv`, `linear_coefficients.csv`, `linear_pred_vs_actual.png`, `linear_residuals.png` |
| `train_ridge.py` | `ridge_metrics.csv`, `ridge_coefficients.csv`, `ridge_alpha_curve.png`, `ridge_coefficient_path.png` |
| `train_random_forest.py` | `rf_grid_search.csv`, `rf_metrics.csv`, `rf_feature_importance.csv`, `rf_pred_vs_actual.png`, `rf_residuals.png` |
| `evaluate_test.py` | `test_metrics.csv`, `test_error_by_co_level.csv`, `test_permutation_importance.csv`, `test_model_comparison.png`, `test_pred_vs_actual.png`, `test_timeline.png`, `test_permutation_importance.png` |

## Team & contributions

| Section | Owner |
|---|---|
| Problem | Ala’a |
| Related Work | Ala’a |
| Dataset | Chris |
| Preprocessing | Chris |
| Methods | Chris |
| Experimental setup | Jeevan |
| Evaluation metrics | Jeevan |
| Results | Jeevan |
| Discussion | Wardah |
| Limitations | Wardah |
| References | — |

## Course

CP322 Final Group Project — code due December 1; presentations December 3–8.
