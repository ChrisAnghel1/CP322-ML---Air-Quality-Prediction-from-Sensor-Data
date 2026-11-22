import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "air_quality_clean.csv"
target = "CO(GT)"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(path, index_col=0, parse_dates=True)
X = df.drop(columns=[target])
y = df[target]

# same chronological split as the linear regression script
n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)
X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)


def score(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# try a range of penalty strengths and keep the one with the lowest validation RMSE
alphas = np.logspace(-3, 3, 61)
val_rmse = []
paths = []
for a in alphas:
    m = Ridge(alpha=a).fit(X_train_s, y_train)
    val_rmse.append(np.sqrt(mean_squared_error(y_val, m.predict(X_val_s))))
    paths.append(m.coef_)
paths = np.array(paths)

best = int(np.argmin(val_rmse))
best_alpha = alphas[best]
print(f"best alpha: {best_alpha:.3f}  (val RMSE {val_rmse[best]:.4f})")
if best in (0, len(alphas) - 1):
    print("warning: best alpha is at the edge of the grid, widen the range")

ridge = Ridge(alpha=best_alpha).fit(X_train_s, y_train)
linear = LinearRegression().fit(X_train_s, y_train)

rows = {
    "linear regression, val": score(y_val, linear.predict(X_val_s)),
    "ridge, train": score(y_train, ridge.predict(X_train_s)),
    "ridge, val": score(y_val, ridge.predict(X_val_s)),
}
metrics = pd.DataFrame(rows).T.round(3)
print("\n", metrics)
metrics.to_csv(f"{out_dir}/ridge_metrics.csv")

coefs = pd.DataFrame({"linear": linear.coef_, "ridge": ridge.coef_}, index=X.columns)
coefs = coefs.reindex(coefs["ridge"].abs().sort_values(ascending=False).index).round(3)
print("\ncoefficients (standardized features):\n", coefs)
coefs.to_csv(f"{out_dir}/ridge_coefficients.csv")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(alphas, val_rmse)
ax.axvline(best_alpha, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("alpha (penalty strength)")
ax.set_ylabel("validation RMSE")
ax.set_title("Ridge: validation error vs alpha")
fig.tight_layout()
fig.savefig(f"{out_dir}/ridge_alpha_curve.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
for i, name in enumerate(X.columns):
    ax.semilogx(alphas, paths[:, i], label=name)
ax.axvline(best_alpha, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("alpha (penalty strength)")
ax.set_ylabel("coefficient")
ax.set_title("Ridge coefficients as the penalty grows")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out_dir}/ridge_coefficient_path.png", dpi=150)