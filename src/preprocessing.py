import sys

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "AirQualityUCI.xlsx"
target = "CO(GT)"

df = pd.read_excel(path)
df["datetime"] = pd.to_datetime(df["Date"].dt.date.astype(str) + " " + df["Time"].astype(str))
df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
df = df.replace(-200, np.nan)
counts = [("start", len(df))]

df = df.drop(columns=["NMHC(GT)"])

# rows where the whole sensor array is down
sensors = [c for c in df.columns if c.startswith("PT08")]
df = df[~df[sensors].isna().all(axis=1)]
counts.append(("sensor outage rows dropped", len(df)))

df = df.dropna(subset=[target])
counts.append(("missing target dropped", len(df)))

# the other GT columns would leak the answer, so they're not features
df = df.drop(columns=[c for c in df.columns if c.endswith("(GT)") and c != target])

prev = None
for label, n in counts:
    print(f"{label}: {n}" + (f" (-{prev - n})" if prev else ""))
    prev = n

print("final shape:", df.shape)
print("remaining NaNs:", int(df.isna().sum().sum()))
df.to_csv("air_quality_clean.csv")