# Data pipeline

## 1. Overview

The pipeline reads the raw UCI Air Quality spreadsheet, turns the -200 missing-value code into real missing values, and drops the rows we cannot use. It keeps CO(GT) as the target and the low-cost sensor and weather columns as features. It writes the result to `air_quality_clean.csv`. There are two scripts in `src/`: `dataset.py` (exploration only, writes nothing) and `preprocessing.py` (produces the clean file).

## 2. Input data

- File: `AirQualityUCI.xlsx` (repo root).
- Size: 9357 rows, 15 columns. One row per hour, from 2004-03-10 to 2005-04-04.
- No duplicate (Date, Time) pairs.

| Group | Columns |
|---|---|
| Timestamp | `Date`, `Time` |
| Reference (GT) analyzers | `CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)` |
| PT08 metal-oxide sensors | `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)` |
| Weather | `T` (temperature), `RH` (relative humidity), `AH` (absolute humidity) |

## 3. Data-quality issues the code handles

**The -200 code.** Missing readings are stored as -200, not blank. Counts of -200 in the raw file: CO(GT) 1683, NOx(GT) 1639, NO2(GT) 1642, NMHC(GT) 8443, and 366 in every other column (PT08, C6H6(GT), T, RH, AH). Both scripts replace -200 with NaN.

**NMHC(GT) is mostly missing.** 8443 of 9357 rows (90.2%) are -200. It cannot be used, so `preprocessing.py` drops the column.

**Sensor outage block.** The 366 rows where a sensor is missing are the same rows for all five PT08 sensors, so the whole sensor array is down at once. (`dataset.py` confirms this by counting rows where every PT08 column is NaN: 366.) T, RH, AH and C6H6(GT) are also missing in 366 rows. These rows have nothing to predict from, so they are dropped.

## 4. Walkthrough, in the order the code runs

Both scripts start the same way. `preprocessing.py` is the one that matters for the report.

### dataset.py (exploration)

1. Load and clean timestamps (lines 4-6). Reads the xlsx, builds a `datetime` column from `Date` and `Time`, drops the two originals, sets `datetime` as the index, sorts by it, and replaces -200 with NaN.
2. Missing percent per column (line 8). Prints the share of NaN in each column. This is where the 90.2% for NMHC(GT) comes from.
3. Outage count (lines 10-12). Picks columns starting with `PT08` and counts rows where all are NaN. Prints 366.
4. Usable rows per target (lines 14-15). For CO(GT), NOx(GT), NO2(GT) and C6H6(GT), counts rows where the target is present and the row is not an outage row. Printed: CO(GT) 7344, NOx(GT) 7396, NO2(GT) 7393, C6H6(GT) 8991. This is how we saw CO(GT) leaves a sensible number of rows.

### preprocessing.py (cleaning)

1. Read input (lines 6-9). Takes the xlsx path from the first command-line argument, default `AirQualityUCI.xlsx`. The target is set to `CO(GT)`.
2. Build the datetime index (lines 10-11). Same as `dataset.py`: combine `Date` and `Time`, drop them, index by datetime, sort.
3. Missing code (line 12). Replace -200 with NaN. Row count is recorded as "start" (line 13).
4. Drop NMHC(GT) (line 15). Too much missing.
5. Drop sensor outage rows (lines 17-20). Removes rows where all PT08 columns are NaN.
6. Drop rows with missing target (lines 22-23). `dropna(subset=["CO(GT)"])`.
7. Drop the other GT columns (lines 25-26). Removes `NOx(GT)`, `NO2(GT)`, `C6H6(GT)`. The code comment says they would leak the answer, so they are not used as features.
8. Report and save (lines 28-35). Prints the row counts, final shape and remaining NaN count, then writes the CSV.

## 5. Row counts

From running `python src/preprocessing.py`:

| Step | Rows left | Rows removed |
|---|---|---|
| Raw file | 9357 | - |
| After dropping sensor outage rows | 8991 | 366 |
| After dropping rows with missing CO(GT) | 7344 | 1647 |

Dropping the NMHC(GT) column and the other GT columns changes columns, not rows. Remaining NaNs in the final data: 0.

## 6. Output

File: `air_quality_clean.csv`, written to the folder the script is run from.

- Shape: 7344 rows by 9 columns, plus the `datetime` index, which is saved as the first CSV column (10 columns in the file).
- Columns: `datetime`, `CO(GT)` (target), `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`.
