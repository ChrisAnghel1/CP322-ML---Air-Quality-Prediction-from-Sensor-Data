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
- Target column: **`CO(GT)`** (set in `src/preprocessing.py`). About 18% of its
  raw values are missing; those rows are dropped.

Full details of the cleaning steps and row counts are in
[docs/data_pipeline.md](docs/data_pipeline.md).

## Preprocessing summary

`src/preprocessing.py` goes from 9,357 raw rows to 7,344 clean rows:

1. Replace `-200` with `NaN`.
2. Drop `NMHC(GT)` (about 90% missing).
3. Drop the 366 sensor-outage rows (all `PT08` columns missing): 8,991 rows left.
4. Drop rows with a missing `CO(GT)`: 7,344 rows left.
5. Drop `NOx(GT)`, `NO2(GT)` and `C6H6(GT)` so they are not used as features.

The output, `air_quality_clean.csv`, has the `datetime` index, the `CO(GT)` target and
8 features (five `PT08.S*` sensors, `T`, `RH`, `AH`), with no missing values.

## Repo structure

```
.
├── AirQualityUCI.xlsx           # raw dataset
├── air_quality_clean.csv        # output of preprocessing.py
├── docs/
│   └── data_pipeline.md         # explanation of the data pipeline
├── src/
│   ├── dataset.py               # exploration: missing %, outage rows, usable rows per target
│   └── preprocessing.py         # cleaning, writes air_quality_clean.csv
├── requirements.txt
└── README.md
```

Modelling and evaluation code is not in the repo yet.

## Setup

```bash
git clone https://github.com/ChrisAnghel1/CP322-ML---Air-Quality-Prediction-from-Sensor-Data.git
cd CP322-ML---Air-Quality-Prediction-from-Sensor-Data
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists `pandas`, `numpy`, `scikit-learn`, `matplotlib` and
`openpyxl` (needed by pandas to read `.xlsx`). The current scripts only use
`pandas`, `numpy` and `openpyxl`.

## Usage

Run both from the repo root, where `AirQualityUCI.xlsx` is:

```bash
python src/dataset.py            # prints missing-data stats
python src/preprocessing.py      # cleans the data, writes air_quality_clean.csv
```

`preprocessing.py` accepts an optional path to the input file as its first argument.

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
