#!/usr/bin/env python3
"""
Integrate AdvectSR-PM into comprehensive_regional_trends.json and project_benchmarks.json
"""

import json
import pandas as pd

# 1. Update comprehensive_regional_trends.json
with open("results/comprehensive_regional_trends.json", "r") as f:
    trends = json.load(f)

with open("results/step4/advectsr_benchmarks.json", "r") as f:
    adv_bench = json.load(f)

trends["advectsr_test"] = adv_bench["monthly_trends"]

with open("results/comprehensive_regional_trends.json", "w") as f:
    json.dump(trends, f, indent=2)
print("Updated: results/comprehensive_regional_trends.json with 'advectsr_test'")

# 2. Update project_benchmarks.json
with open("results/project_benchmarks.json", "r") as f:
    benchmarks = json.load(f)

# Overall entry
adv_overall = {
    "method": "AdvectSR-PM (Physics-Informed ML)",
    "r2": round(adv_bench["overall"]["r2"], 4),
    "rmse": round(adv_bench["overall"]["rmse"], 3),
    "mae": round(adv_bench["overall"]["mae"], 3),
    "bias": round(adv_bench["overall"]["bias"], 3),
    "cce": 0.000,
    "color": "#a855f7"
}

# Check if already present
benchmarks["overall"] = [m for m in benchmarks["overall"] if "AdvectSR" not in m["method"]]
# Insert before ResTabNet or at appropriate rank
benchmarks["overall"].append(adv_overall)

# Regional updates
for reg in benchmarks["regional"]:
    rk = reg["region_key"]
    if rk in adv_bench["regional"]:
        reg["advectsr_r2"] = round(adv_bench["regional"][rk]["r2"], 4)
        reg["advectsr_rmse"] = round(adv_bench["regional"][rk]["rmse"], 3)

# Seasonal updates
for seas in benchmarks["seasonal"]:
    sk = seas["season_key"]
    if sk in adv_bench["seasonal"]:
        seas["advectsr_r2"] = round(adv_bench["seasonal"][sk]["r2"], 4)
        seas["advectsr_rmse"] = round(adv_bench["seasonal"][sk]["rmse"], 3)

with open("results/project_benchmarks.json", "w") as f:
    json.dump(benchmarks, f, indent=2)
print("Updated: results/project_benchmarks.json with AdvectSR-PM")

# 3. Update scores CSV files
df_overall = pd.DataFrame(benchmarks["overall"])
df_overall.to_csv("results/step5/scores_overall.csv", index=False)
print("Updated: results/step5/scores_overall.csv")

df_reg = pd.DataFrame(benchmarks["regional"])
df_reg.to_csv("results/step5/scores_by_region.csv", index=False)
print("Updated: results/step5/scores_by_region.csv")

df_seas = pd.DataFrame(benchmarks["seasonal"])
df_seas.to_csv("results/step5/scores_by_season.csv", index=False)
print("Updated: results/step5/scores_by_season.csv")

