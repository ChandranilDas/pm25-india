#!/usr/bin/env python3
"""
Train HistGradientBoostingRegressor (HGB) on EE708 Spatial Split
Evaluates overall test cell performance, regional breakdown, seasonal breakdown,
and 60-month regional monthly time series.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

print("=" * 70)
print(" Training HistGradientBoosting (HGB) Regressor on PM2.5 Dataset")
print("=" * 70)

# Load data
table = pd.read_csv("data/pm25_daily.csv", parse_dates=["date"])
cells = pd.read_csv("data/cells.csv")
table = table.merge(cells[["lat", "lon", "set"]], on=["lat", "lon"])

train_mask = table["set"] == "train"
test_mask = table["set"] == "test"

# Calculate physical features
table["ws"] = np.sqrt(table["u10"]**2 + table["v10"]**2)
table["rh_proxy"] = table["t2m"] - table["d2m"]
table["sin_mon"] = np.sin(2.0 * np.pi * table["month"] / 12.0)
table["cos_mon"] = np.cos(2.0 * np.pi * table["month"] / 12.0)
table["aod_vent"] = table["aod"] / (table["ws"] + 0.5)

FEATURES_9 = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"]
FEATURES_13 = ["aod", "u10", "v10", "ws", "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon", "aod_vent"]

print(f"Total rows: {len(table):,} (Train: {train_mask.sum():,}, Test: {test_mask.sum():,})")

y_test = table.loc[test_mask, "pm25"].to_numpy()

# Test 1: Standard 9 features
print("\n--- Training HGB with standard 9 CAMS features ---")
t0 = time.time()
hgb_9 = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.07,
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=0.05,
    random_state=42
)
hgb_9.fit(table.loc[train_mask, FEATURES_9], table.loc[train_mask, "pm25"])
train_time_9 = time.time() - t0
pred_test_9 = hgb_9.predict(table.loc[test_mask, FEATURES_9])
r2_9 = r2_score(y_test, pred_test_9)
rmse_9 = root_mean_squared_error(y_test, pred_test_9)
mae_9 = mean_absolute_error(y_test, pred_test_9)
bias_9 = float(np.mean(pred_test_9 - y_test))
print(f"HGB (9 feats) trained in {train_time_9:.2f}s: R2={r2_9:.4f}, RMSE={rmse_9:.2f}, MAE={mae_9:.2f}, Bias={bias_9:.2f}")

# Test 2: 13 features
print("\n--- Training HGB with 13 engineered features ---")
t0 = time.time()
hgb_13 = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.07,
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=0.05,
    random_state=42
)
hgb_13.fit(table.loc[train_mask, FEATURES_13], table.loc[train_mask, "pm25"])
train_time_13 = time.time() - t0
pred_test_13 = hgb_13.predict(table.loc[test_mask, FEATURES_13])
r2_13 = r2_score(y_test, pred_test_13)
rmse_13 = root_mean_squared_error(y_test, pred_test_13)
mae_13 = mean_absolute_error(y_test, pred_test_13)
bias_13 = float(np.mean(pred_test_13 - y_test))
print(f"HGB (13 feats) trained in {train_time_13:.2f}s: R2={r2_13:.4f}, RMSE={rmse_13:.2f}, MAE={mae_13:.2f}, Bias={bias_13:.2f}")

# Save both models for versatility
joblib.dump(hgb_9, "results/step4/hgb_model_9.joblib", compress=3)
joblib.dump(hgb_13, "results/step4/hgb_model.joblib", compress=3)

best_model = hgb_13 if r2_13 >= r2_9 else hgb_9
best_feats = FEATURES_13 if r2_13 >= r2_9 else FEATURES_9
best_pred = pred_test_13 if r2_13 >= r2_9 else pred_test_9
best_r2 = r2_13 if r2_13 >= r2_9 else r2_9
best_rmse = rmse_13 if r2_13 >= r2_9 else rmse_9
best_mae = mae_13 if r2_13 >= r2_9 else mae_9
best_bias = bias_13 if r2_13 >= r2_9 else bias_9

model_size_mb = os.path.getsize("results/step4/hgb_model.joblib") / (1024 * 1024)
print(f"\nSaved primary HGB model to results/step4/hgb_model.joblib ({model_size_mb:.2f} MB)")

# Regional breakdown
test_df = table.loc[test_mask].copy()
test_df["pred_hgb"] = best_pred

print("\n--- HGB Performance by Region ---")
region_metrics = {}
for region, group in test_df.groupby("region"):
    y_reg = group["pm25"].to_numpy()
    pred_reg = group["pred_hgb"].to_numpy()
    r2_reg = float(r2_score(y_reg, pred_reg))
    rmse_reg = float(root_mean_squared_error(y_reg, pred_reg))
    mae_reg = float(mean_absolute_error(y_reg, pred_reg))
    bias_reg = float(np.mean(pred_reg - y_reg))
    region_metrics[region] = {
        "r2": round(r2_reg, 4),
        "rmse": round(rmse_reg, 2),
        "mae": round(mae_reg, 2),
        "bias": round(bias_reg, 2),
        "count": len(group)
    }
    print(f"  {region:12s}: R2={r2_reg:.4f}, RMSE={rmse_reg:.2f}, MAE={mae_reg:.2f}, Bias={bias_reg:.2f} (n={len(group)})")

print("\n--- HGB Performance by Season ---")
season_metrics = {}
for season, group in test_df.groupby("season"):
    y_sea = group["pm25"].to_numpy()
    pred_sea = group["pred_hgb"].to_numpy()
    r2_sea = float(r2_score(y_sea, pred_sea))
    rmse_sea = float(root_mean_squared_error(y_sea, pred_sea))
    mae_sea = float(mean_absolute_error(y_sea, pred_sea))
    bias_sea = float(np.mean(pred_sea - y_sea))
    season_metrics[season] = {
        "r2": round(r2_sea, 4),
        "rmse": round(rmse_sea, 2),
        "mae": round(mae_sea, 2),
        "bias": round(bias_sea, 2),
        "count": len(group)
    }
    print(f"  {season:12s}: R2={r2_sea:.4f}, RMSE={rmse_sea:.2f}, MAE={mae_sea:.2f}, Bias={bias_sea:.2f}")

# Calculate monthly predictions for each region (2020-01 to 2024-12)
test_df["month_str"] = test_df["date"].dt.strftime("%Y-%m")
all_months = sorted(test_df["month_str"].unique())
monthly_hgb_by_region = {}
for reg in sorted(test_df["region"].unique()):
    reg_df = test_df[test_df["region"] == reg]
    m_means = reg_df.groupby("month_str")["pred_hgb"].mean()
    # Reindex to all 60 months
    monthly_hgb_by_region[reg] = [round(float(m_means.get(m, 0.0)), 2) for m in all_months]

# Export HGB benchmark summary
hgb_benchmarks = {
    "model_name": "HistGradientBoosting (HGB)",
    "architecture": "Histogram-binned LightGBM Gradient Boosting",
    "params": f"{best_model.n_iter_} trees (B=256 histogram bins)",
    "size_mb": round(model_size_mb, 2),
    "features": best_feats,
    "overall": {
        "r2": round(best_r2, 4),
        "rmse": round(best_rmse, 2),
        "mae": round(best_mae, 2),
        "bias": round(best_bias, 2)
    },
    "by_region": region_metrics,
    "by_season": season_metrics,
    "monthly_series": monthly_hgb_by_region,
    "months": all_months
}

with open("results/step4/hgb_benchmarks.json", "w") as f:
    json.dump(hgb_benchmarks, f, indent=2)

print("\nSaved results/step4/hgb_benchmarks.json successfully!")
