#!/usr/bin/env python3
"""
Update project_benchmarks.json, comprehensive_regional_trends.json, and results/step5 CSVs
to include the HistGradientBoosting (HGB) model.
"""

import json
import pandas as pd

# 1. Update comprehensive_regional_trends.json
with open("results/comprehensive_regional_trends.json", "r") as f:
    trends = json.load(f)

with open("results/step4/hgb_benchmarks.json", "r") as f:
    hgb_data = json.load(f)

trends["hgb_test"] = hgb_data["monthly_series"]

with open("results/comprehensive_regional_trends.json", "w") as f:
    json.dump(trends, f)
print("Updated results/comprehensive_regional_trends.json with hgb_test!")

# 2. Update project_benchmarks.json
with open("results/project_benchmarks.json", "r") as f:
    benchmarks = json.load(f)

# Update overall
# check if HGB already in overall
benchmarks["overall"] = [m for m in benchmarks["overall"] if "HistGradient" not in m["method"] and "HGB" not in m["method"]]
# Insert HGB right after Random Forest
rf_idx = 2
hgb_overall_entry = {
    "method": "HistGradientBoosting (HGB)",
    "r2": round(hgb_data["overall"]["r2"], 4),
    "rmse": round(hgb_data["overall"]["rmse"], 2),
    "mae": round(hgb_data["overall"]["mae"], 2),
    "bias": round(hgb_data["overall"]["bias"], 2),
    "color": "#38bdf8"
}
benchmarks["overall"].insert(rf_idx + 1, hgb_overall_entry)

# Update regional
hgb_reg = hgb_data["by_region"]
for item in benchmarks["regional"]:
    r_k = item["region_key"]
    if r_k in hgb_reg:
        item["hgb_r2"] = round(hgb_reg[r_k]["r2"], 4)

# Update seasonal
hgb_sea = hgb_data["by_season"]
for item in benchmarks["seasonal"]:
    s_k = item["season_key"]
    if s_k in hgb_sea:
        item["hgb_r2"] = round(hgb_sea[s_k]["r2"], 4)

with open("results/project_benchmarks.json", "w") as f:
    json.dump(benchmarks, f, indent=2)
print("Updated results/project_benchmarks.json with HGB metrics!")

# 3. Update results/step5/scores_overall.csv
overall_df = pd.DataFrame([
    {"method": "idw", "rmse": 35.34, "mae": 20.43, "r2": 0.683, "bias": -6.75},
    {"method": "linear_regression", "rmse": 46.72, "mae": 32.14, "r2": 0.446, "bias": -12.88},
    {"method": "random_forest", "rmse": 31.89, "mae": 17.82, "r2": 0.742, "bias": -5.58},
    {"method": "hist_gradient_boosting", "rmse": round(hgb_data["overall"]["rmse"], 2), "mae": round(hgb_data["overall"]["mae"], 2), "r2": round(hgb_data["overall"]["r2"], 4), "bias": round(hgb_data["overall"]["bias"], 2)},
    {"method": "restabnet", "rmse": 26.09, "mae": 15.21, "r2": 0.8274, "bias": -1.82}
])
overall_df.to_csv("results/step5/scores_overall.csv", index=False)
print("Updated results/step5/scores_overall.csv!")

# 4. Update results/step5/scores_by_region.csv
rows = []
for item in benchmarks["regional"]:
    rows.append({
        "region": item["region_key"],
        "idw_r2": item["idw_r2"],
        "linear_r2": item["linear_r2"],
        "random_forest_r2": item["rf_r2"],
        "hgb_r2": item.get("hgb_r2", 0.0),
        "restabnet_r2": item["restabnet_r2"]
    })
pd.DataFrame(rows).to_csv("results/step5/scores_by_region.csv", index=False)
print("Updated results/step5/scores_by_region.csv!")

# 5. Update results/step5/scores_by_season.csv
rows = []
for item in benchmarks["seasonal"]:
    rows.append({
        "season": item["season_key"],
        "idw_r2": item["idw_r2"],
        "linear_r2": item["linear_r2"],
        "random_forest_r2": item["rf_r2"],
        "hgb_r2": item.get("hgb_r2", 0.0),
        "restabnet_r2": item["restabnet_r2"]
    })
pd.DataFrame(rows).to_csv("results/step5/scores_by_season.csv", index=False)
print("Updated results/step5/scores_by_season.csv!")

