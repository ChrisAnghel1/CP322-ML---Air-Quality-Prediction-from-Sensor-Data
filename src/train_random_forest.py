import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = sys.argv[1] if len(sys.argv) > 1 else "air_quality_clean.csv"
target = "CO(GT)"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(path, index_col=0, parse_dates=True)
X = df.drop(columns=[target])
y = df[target]

# chronological split, same as linear and ridge
n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)
X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
print(f"train {len(X_train)}, val {len(X_val)}, test {n - i_val}")

def score(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# grid search on the validation set
grid = {
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 5, 20, 50, 100],
}
search = []
for depth, leaf in product(grid["max_depth"], grid["min_samples_leaf"]):
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
        random_state=0, n_jobs=-1,
    ).fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
    search.append({"max_depth": depth, "min_samples_leaf": leaf, "val_RMSE": round(rmse, 4)})
    print(f"max_depth={depth}, min_samples_leaf={leaf}: val RMSE {rmse:.4f}")

search = pd.DataFrame(search)
search.to_csv(f"{out_dir}/rf_grid_search.csv", index=False)
best = search.loc[search["val_RMSE"].idxmin()]
best_depth = None if pd.isna(best["max_depth"]) else int(best["max_depth"])
best_leaf = int(best["min_samples_leaf"])
print(f"\nbest: max_depth={best_depth}, min_samples_leaf={best_leaf}")

model = RandomForestRegressor(
    n_estimators=200, max_depth=best_depth, min_samples_leaf=best_leaf,
    random_state=0, n_jobs=-1,
).fit(X_train, y_train)

val_pred = model.predict(X_val)
rows = {
    "random forest, train": score(y_train, model.predict(X_train)),
    "random forest, val": score(y_val, val_pred),
}
metrics = pd.DataFrame(rows).T.round(3)
print("\n", metrics)
metrics.to_csv(f"{out_dir}/rf_metrics.csv")

# feature importances, sorted by size
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).round(3)
print("\nfeature importances:\n", importances)
importances.rename("importance").to_csv(f"{out_dir}/rf_feature_importance.csv")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_val, val_pred, s=5, alpha=0.4)
lims = [min(y_val.min(), val_pred.min()), max(y_val.max(), val_pred.max())]
ax.plot(lims, lims, color="black", linewidth=1)
ax.set_xlabel("actual CO(GT)")
ax.set_ylabel("predicted CO(GT)")
ax.set_title("Random forest, validation set")
fig.tight_layout()
fig.savefig(f"{out_dir}/rf_pred_vs_actual.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(y_val.index, y_val - val_pred, linewidth=0.5)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("residual")
ax.set_title("Random forest residuals over time, validation set")
fig.tight_layout()
fig.savefig(f"{out_dir}/rf_residuals.png", dpi=150)
