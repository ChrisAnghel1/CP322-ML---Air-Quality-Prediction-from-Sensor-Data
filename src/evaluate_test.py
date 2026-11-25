"""Final comparison of all models on the held-out test set.

Uses the same chronological 70/15/15 split as the training scripts.
Each model's settings are chosen on the validation set only, then the
model is refit on the training set and scored once on the test set.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
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

n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)
X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
X_test, y_test = X.iloc[i_val:], y.iloc[i_val:]
for name, part in [("train", X_train), ("val", X_val), ("test", X_test)]:
    print(f"{name}: {len(part)} rows, {part.index.min():%Y-%m-%d} to {part.index.max():%Y-%m-%d}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = (scaler.transform(a) for a in (X_train, X_val, X_test))


def score(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ridge: same alpha grid as train_ridge.py, picked on validation
alphas = np.logspace(-3, 3, 61)
val_scores = [rmse(y_val, Ridge(alpha=a).fit(X_train_s, y_train).predict(X_val_s)) for a in alphas]
best_alpha = alphas[int(np.argmin(val_scores))]

# random forest: same grid as train_random_forest.py, picked on validation
best_rf, best_rf_rmse = None, np.inf
for depth in [5, 10, 15, None]:
    for leaf in [1, 5, 20, 50, 100]:
        m = RandomForestRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                  random_state=0, n_jobs=-1).fit(X_train, y_train)
        s = rmse(y_val, m.predict(X_val))
        if s < best_rf_rmse:
            best_rf, best_rf_rmse = (depth, leaf), s
print(f"ridge alpha={best_alpha:.2f}, rf max_depth={best_rf[0]}, min_samples_leaf={best_rf[1]}")

models = {
    "Mean baseline": None,
    "Linear regression": LinearRegression().fit(X_train_s, y_train),
    "Ridge": Ridge(alpha=best_alpha).fit(X_train_s, y_train),
    "Random forest": RandomForestRegressor(n_estimators=200, max_depth=best_rf[0],
                                           min_samples_leaf=best_rf[1], random_state=0,
                                           n_jobs=-1).fit(X_train, y_train),
}
uses_scaled = {"Linear regression", "Ridge"}

preds, rows = {}, []
for name, m in models.items():
    if m is None:
        p_val = np.full(len(y_val), y_train.mean())
        p_test = np.full(len(y_test), y_train.mean())
    elif name in uses_scaled:
        p_val, p_test = m.predict(X_val_s), m.predict(X_test_s)
    else:
        p_val, p_test = m.predict(X_val), m.predict(X_test)
    preds[name] = p_test
    for split, yt, p in [("validation", y_val, p_val), ("test", y_test, p_test)]:
        rows.append({"model": name, "split": split, **score(yt, p)})

metrics = pd.DataFrame(rows).round(3)
print("\n", metrics.to_string(index=False))
metrics.to_csv(f"{out_dir}/test_metrics.csv", index=False)

# error by CO level on the test set
bins = [0, 1, 2, 4, np.inf]
labels = ["0-1", "1-2", "2-4", "4+"]
band = pd.cut(y_test, bins=bins, labels=labels, right=False)
band_rows = []
for name in ["Linear regression", "Ridge", "Random forest"]:
    err = pd.Series(np.abs(y_test.values - preds[name]), index=y_test.index)
    bias = pd.Series(preds[name] - y_test.values, index=y_test.index)
    for lab in labels:
        mask = (band == lab).values
        band_rows.append({"model": name, "CO band (mg/m3)": lab, "rows": int(mask.sum()),
                          "MAE": err[mask].mean(), "mean bias": bias[mask].mean()})
bands = pd.DataFrame(band_rows).round(3)
print("\n", bands.to_string(index=False))
bands.to_csv(f"{out_dir}/test_error_by_co_level.csv", index=False)

# permutation importance on the test set (model-agnostic, comparable across models)
imp = {}
for name in ["Linear regression", "Random forest"]:
    Xt = X_test_s if name in uses_scaled else X_test
    r = permutation_importance(models[name], Xt, y_test, n_repeats=20, random_state=0,
                               scoring="neg_root_mean_squared_error")
    imp[name] = r.importances_mean
imp = pd.DataFrame(imp, index=X.columns).sort_values("Random forest", ascending=False).round(3)
imp.columns = [f"{c} (RMSE increase)" for c in imp.columns]
print("\n", imp)
imp.to_csv(f"{out_dir}/test_permutation_importance.csv")

# figures
best_name = metrics[metrics.split == "test"].set_index("model")["RMSE"].drop("Mean baseline").idxmin()

t = metrics[(metrics.split == "test") & (metrics.model != "Mean baseline")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, col in zip(axes, ["RMSE", "MAE", "R2"]):
    ax.bar(t.model, t[col], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title(f"Test {col}")
    for i, v in enumerate(t[col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.margins(y=0.15)
fig.tight_layout()
fig.savefig(f"{out_dir}/test_model_comparison.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, preds[best_name], s=5, alpha=0.4)
lims = [0, max(y_test.max(), preds[best_name].max())]
ax.plot(lims, lims, color="black", linewidth=1)
ax.set_xlabel("actual CO(GT) (mg/m³)")
ax.set_ylabel("predicted CO(GT) (mg/m³)")
ax.set_title(f"{best_name}, test set")
fig.tight_layout()
fig.savefig(f"{out_dir}/test_pred_vs_actual.png", dpi=150)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(y_test.index, y_test.values, linewidth=0.8, label="actual", color="black")
ax.plot(y_test.index, preds[best_name], linewidth=0.8, label=f"{best_name}", color="#C44E52")
ax.set_ylabel("CO(GT) (mg/m³)")
ax.set_title("Test set over time")
ax.legend()
fig.tight_layout()
fig.savefig(f"{out_dir}/test_timeline.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
imp_plot = imp.iloc[::-1]
yy = np.arange(len(imp_plot))
ax.barh(yy - 0.2, imp_plot.iloc[:, 0], height=0.4, label="Linear regression")
ax.barh(yy + 0.2, imp_plot.iloc[:, 1], height=0.4, label="Random forest")
ax.set_yticks(yy, imp_plot.index)
ax.set_xlabel("increase in test RMSE when feature is shuffled")
ax.set_title("Permutation feature importance, test set")
ax.legend()
fig.tight_layout()
fig.savefig(f"{out_dir}/test_permutation_importance.png", dpi=150)
print("\nbest model on test:", best_name)
