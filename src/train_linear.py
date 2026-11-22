import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "air_quality_clean.csv"
target = "CO(GT)"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(path, index_col=0, parse_dates=True)
X = df.drop(columns=[target])
y = df[target]

# chronological split, no shuffling. test set stays untouched until the end
n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)
X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
print(f"train {len(X_train)}, val {len(X_val)}, test {n - i_val}")

# scaler is fitted on the training rows only
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)

model = LinearRegression().fit(X_train_s, y_train)


def score(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# naive baseline: predict the training mean every time
base_val = np.full(len(y_val), y_train.mean())

rows = {
    "linear regression, train": score(y_train, model.predict(X_train_s)),
    "linear regression, val": score(y_val, model.predict(X_val_s)),
    "mean baseline, val": score(y_val, base_val),
}
metrics = pd.DataFrame(rows).T.round(3)
print("\n", metrics)
metrics.to_csv(f"{out_dir}/linear_metrics.csv")

# standardized coefficients, sorted by size
coefs = pd.Series(model.coef_, index=X.columns)
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(3)
print("\ncoefficients (standardized features):\n", coefs)
coefs.rename("coefficient").to_csv(f"{out_dir}/linear_coefficients.csv")

# how correlated the sensors are with each other
print("\nfeature correlations:\n", X_train.corr().round(2))

val_pred = model.predict(X_val_s)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_val, val_pred, s=5, alpha=0.4)
lims = [min(y_val.min(), val_pred.min()), max(y_val.max(), val_pred.max())]
ax.plot(lims, lims, color="black", linewidth=1)
ax.set_xlabel("actual CO(GT)")
ax.set_ylabel("predicted CO(GT)")
ax.set_title("Linear regression, validation set")
fig.tight_layout()
fig.savefig(f"{out_dir}/linear_pred_vs_actual.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(y_val.index, y_val - val_pred, linewidth=0.5)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("residual")
ax.set_title("Linear regression residuals over time, validation set")
fig.tight_layout()
fig.savefig(f"{out_dir}/linear_residuals.png", dpi=150)

print(f"\nsaved metrics, coefficients and two plots to {out_dir}/")