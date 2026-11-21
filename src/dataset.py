import pandas as pd
import numpy as np

df = pd.read_excel("AirQualityUCI.xlsx")
df["datetime"] = pd.to_datetime(df["Date"].dt.date.astype(str) + " " + df["Time"].astype(str))
df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index().replace(-200, np.nan)

print(df.isna().mean().sort_values(ascending=False) * 100) # missing % per column

sensors = [c for c in df.columns if c.startswith("PT08")]
outage = df[sensors].isna().all(axis=1)
print("All-sensor outage rows:", outage.sum()) # expect 366

for t in ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]: # usable rows per target
    print(t, (df[t].notna() & ~outage).sum())