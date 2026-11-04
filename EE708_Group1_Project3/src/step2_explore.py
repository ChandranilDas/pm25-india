# Step 2: Look at the data before modelling
#
# - Average PM2.5 by region and by season (table).
# - Correlation of each predictor with PM2.5 (table).
# - Map of average PM2.5 and a plot of monthly PM2.5 over the 5 years.
#
# Run: conda run -n pm25 python step2_explore.py

import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/step2", exist_ok=True)
table = pd.read_csv("data/pm25_daily.csv", parse_dates=["date"])

# 1. Average PM2.5 for every region and season.
region_season = table.pivot_table(index="region", columns="season", values="pm25")
region_season = region_season[["winter", "pre_monsoon", "monsoon", "post_monsoon"]]
print("Average PM2.5 (µg/m³) by region and season:")
print(region_season.round(1))
region_season.round(1).to_csv("results/step2/pm25_by_region_season.csv")

# 2. Correlation of each predictor with PM2.5.
predictors = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon"]
correlation = table[predictors].corrwith(table["pm25"]).sort_values()
print("\nCorrelation with PM2.5:")
print(correlation.round(2))
correlation.round(3).to_csv("results/step2/correlation_with_pm25.csv", header=["r"])

# 3. Map of average PM2.5 over the 5 years.
cell_mean = table.groupby(["lat", "lon"])["pm25"].mean().reset_index()
plt.figure(figsize=(7, 7))
plt.scatter(cell_mean["lon"], cell_mean["lat"], c=cell_mean["pm25"],
            cmap="YlOrRd", marker="s", s=60)
plt.colorbar(label="PM2.5 (µg/m³)")
plt.title("Average PM2.5, 2020-2024")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("results/step2/map_mean_pm25.png", dpi=120, bbox_inches="tight")
plt.close()

# 4. Monthly average PM2.5 for each region.
monthly = table.groupby([pd.Grouper(key="date", freq="MS"), "region"])["pm25"].mean()
monthly.unstack().plot(figsize=(11, 4))
plt.ylabel("PM2.5 (µg/m³)")
plt.title("Monthly average PM2.5 by region")
plt.savefig("results/step2/monthly_pm25_by_region.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nFigures and tables saved in results/step2/")
