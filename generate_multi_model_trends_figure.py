#!/usr/bin/env python3
"""
Generate publication-grade monthly trend figures:
  1. results/step2/monthly_pm25_by_region.png (high-res 6-region ground truth)
  2. results/step2/monthly_pm25_triple_comparison.png (6-panel subplots: Ground Truth vs ResTabNet vs HGB vs RF)
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/.mpl"

os.makedirs("results/step2", exist_ok=True)

# 1. Regenerate crisp Ground Truth line plot (Image 1)
df = pd.read_csv("data/pm25_daily.csv", parse_dates=["date"])
monthly = df.groupby([pd.Grouper(key="date", freq="MS"), "region"])["pm25"].mean().unstack()

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
colors = {
    "central": "#1f77b4",
    "himalaya": "#ff7f0e",
    "igp": "#2ca02c",
    "northeast": "#d62728",
    "northwest": "#9467bd",
    "south": "#8c564b"
}
for reg in ["central", "himalaya", "igp", "northeast", "northwest", "south"]:
    if reg in monthly.columns:
        ax.plot(monthly.index, monthly[reg], label=reg, color=colors.get(reg, None), linewidth=1.8)

ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11, fontweight="bold")
ax.set_xlabel("date", fontsize=11, fontweight="bold")
ax.set_title("Monthly average PM2.5 by region", fontsize=13, fontweight="bold")
ax.set_xlim(monthly.index.min(), monthly.index.max())
ax.legend(title="region", loc="upper right", framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.5)
plt.savefig("results/step2/monthly_pm25_by_region.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/step2/monthly_pm25_by_region.png")


# 2. Multi-Model Regional Trends Subplots (6 regions)
with open("results/comprehensive_regional_trends.json", "r") as f:
    trends = json.load(f)

dates = pd.to_datetime(trends["dates"])

region_titles = {
    "igp": "Indo-Gangetic Plain (IGP)",
    "central": "Central India",
    "northwest": "Northwest (Arid)",
    "south": "South (Peninsular)",
    "northeast": "Northeast (Valley)",
    "himalaya": "Himalayan Basin"
}

fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=150, sharex=True)
axes = axes.flatten()

for idx, reg_key in enumerate(["igp", "central", "northwest", "south", "northeast", "himalaya"]):
    ax = axes[idx]
    
    true_vals = trends["observed_test"][reg_key]
    res_vals = trends["restabnet_test"][reg_key]
    adv_vals = trends["advectsr_test"][reg_key]
    hgb_vals = trends["hgb_test"][reg_key]
    rf_vals = trends["random_forest_test"][reg_key]
    
    ax.plot(dates, true_vals, label="CAMS Ground Truth", color="#22c55e", linewidth=2.2)
    ax.plot(dates, res_vals, label="ResTabNet (R²=0.827)", color="#38bdf8", linestyle="-", linewidth=2.0)
    ax.plot(dates, adv_vals, label="AdvectSR-PM (R²=0.777)", color="#a855f7", linestyle="--", linewidth=1.9)
    ax.plot(dates, hgb_vals, label="HistGradBoost (R²=0.754)", color="#eab308", linestyle="-.", linewidth=1.7)
    ax.plot(dates, rf_vals, label="Random Forest (R²=0.742)", color="#f97316", linestyle=":", linewidth=1.7)
    
    ax.set_title(region_titles.get(reg_key, reg_key), fontsize=12, fontweight="bold")
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.autofmt_xdate()
fig.suptitle("Monthly Average PM2.5 (2020–2024): Multi-Model Architecture Tracking across 6 Geographical Zones", fontsize=14, fontweight="bold", y=0.98)
plt.savefig("results/step2/monthly_pm25_triple_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/step2/monthly_pm25_triple_comparison.png")
