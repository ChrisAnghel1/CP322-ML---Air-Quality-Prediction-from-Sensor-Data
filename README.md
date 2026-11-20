# CP322 Final Group Project — Air Quality Prediction

Regression project for CP322: predicting ground-truth pollutant concentration from
low-cost metal-oxide sensor readings, using the UCI Air Quality dataset.

## Problem

Given hourly readings from an array of five metal-oxide chemical sensors plus
temperature/humidity, predict the concentration of a target pollutant as measured
by a co-located reference analyzer (the dataset's `(GT)` columns). Task type:
regression.

## Dataset

- Source: [UCI Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality)
- 9,357 hourly records, March 2004 – April 2005, from a polluted area in an Italian city
- 15 columns: `Date`, `Time`, five reference-analyzer `(GT)` concentrations
  (`CO(GT)`, `NMHC(GT)`, `NOx(GT)`, `NO2(GT)`), one derived hydrocarbon reading
  (`C6H6(GT)`), five `PT08.S*` metal-oxide sensor responses, and `T`/`RH`/`AH`
  (temperature, relative humidity, absolute humidity)
- **Use `AirQualityUCI.xlsx`, not the `.csv`.** The CSV uses comma decimals and
  has trailing empty columns; the xlsx is clean.
- Missing readings are coded as the literal value `-200`, not a blank cell —
  this must be replaced with `NaN` before any analysis.
- `NMHC(GT)` is missing in ~90% of rows, despite the UCI documentation listing
  it as complete. It is dropped as unusable rather than imputed.
- A block of 366 rows has an entire group of sensor columns (`PT08.S1(CO)`,
  `C6H6(GT)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`,
  `T`, `RH`, `AH`) missing simultaneously — a single sensor-array outage rather
  than scattered gaps, and handled separately from per-channel missingness.
- Target column: **TBD** — candidates are `CO(GT)`, `NOx(GT)`, or `NO2(GT)`
  (each ~17–18% missing, far more usable than `NMHC(GT)`).

## Repo structure

```
.
├── data/
│   └── AirQualityUCI.xlsx       # raw dataset (not committed if large / licensed)
├── src/
│   ├── preprocessing.py         # loading, -200 handling, missing-data strategy
│   ├── features.py              # feature engineering
│   ├── train.py                 # model training
│   └── evaluate.py              # metrics, feature importance
├── notebooks/                   # exploratory analysis
├── report/                      # final written report
└── README.md
```

*(Adjust to match the actual layout once the code is organized.)*

## Setup

```bash
git clone <https://github.com/ChrisAnghel1/CP322-ML---Air-Quality-Prediction-from-Sensor-Data.git>
cd <CP322-ML Air Quality Predictions from Sensor Data>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Suggested `requirements.txt` contents: `pandas`, `numpy`, `scikit-learn`,
`matplotlib`, `openpyxl` (needed by pandas to read `.xlsx`).

## Usage

```bash
python src/preprocessing.py      # cleans raw data, writes processed dataset
python src/train.py              # trains model(s)
python src/evaluate.py           # reports metrics + feature importance
```

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
