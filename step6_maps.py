# Step 6: Finer PM2.5 maps (0.25° instead of the CAMS 0.75°)
#
# Two ways to make a finer map:
#   - Standard interpolation: interpolate CAMS PM2.5 itself to the 0.25° points
#     (bilinear interpolation).
#   - ML-based: interpolate the predictors (AOD, wind, temperature, ...) to the
#     0.25° points, then let the random forest predict PM2.5 there.
# We make the maps for one example day and for the 2024 average.
#
# Run: conda run -n pm25 python step6_maps.py

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

os.makedirs("results/step6", exist_ok=True)
FEATURES = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"]
EXAMPLE_DAY = "2024-11-15"

data = xr.open_dataset("data/pm25_daily.nc")
data = data.rename({"pm2p5": "pm25", "aod550": "aod"})
forest = joblib.load("results/step4/random_forest.joblib")

# The 0.25° grid (3 fine points per CAMS cell in each direction).
fine_lat = np.arange(data["lat"].min(), data["lat"].max() + 0.01, 0.25)
fine_lon = np.arange(data["lon"].min(), data["lon"].max() + 0.01, 0.25)
# A fine point is inside India when its nearest CAMS cell is inside India.
fine_in_india = data["in_india"].astype(int).interp(lat=fine_lat, lon=fine_lon,
                                                    method="nearest") == 1


def fine_maps(first_day, last_day):
    """Bilinear and random forest PM2.5 on the 0.25° grid, averaged over the days."""
    days = data[["aod", "u10", "v10", "t2m", "d2m", "sp", "pm25"]].sel(time=slice(first_day, last_day))
    fine = days.interp(lat=fine_lat, lon=fine_lon)  # bilinear interpolation
    table = fine.to_dataframe().reset_index()
    table["month"] = table["time"].dt.month
    table["random_forest"] = forest.predict(table[FEATURES])
    table = table.rename(columns={"pm25": "bilinear"})
    mean = table.groupby(["lat", "lon"])[["bilinear", "random_forest"]].mean().to_xarray()
    return mean.where(fine_in_india)


def plot(maps, title, file_name, vmax):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (name, values) in zip(axes, maps.items()):
        im = ax.pcolormesh(values["lon"], values["lat"], values, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes, shrink=0.8, label="PM2.5 (µg/m³)")
    fig.suptitle(title)
    fig.savefig(file_name, dpi=120, bbox_inches="tight")
    plt.close(fig)


# 1. One example day.
day = fine_maps(EXAMPLE_DAY, EXAMPLE_DAY)
cams_day = data["pm25"].sel(time=EXAMPLE_DAY).where(data["in_india"])
plot({"CAMS 0.75°": cams_day,
      "Bilinear interpolation 0.25°": day["bilinear"],
      "Random forest 0.25°": day["random_forest"]},
     f"PM2.5 on {EXAMPLE_DAY}", "results/step6/map_example_day.png", vmax=400)

# 2. Average of 2024.
year = fine_maps("2024-01-01", "2024-12-31")
cams_year = data["pm25"].sel(time="2024").mean("time").where(data["in_india"])
plot({"CAMS 0.75°": cams_year,
      "Bilinear interpolation 0.25°": year["bilinear"],
      "Random forest 0.25°": year["random_forest"]},
     "Average PM2.5 in 2024", "results/step6/map_2024_mean.png", vmax=200)

# 3. Simple check: average over India of each 2024 map, and how different the
# two fine maps are.
check = pd.Series({
    "CAMS 0.75° mean": float(cams_year.mean()),
    "Bilinear 0.25° mean": float(year["bilinear"].mean()),
    "Random forest 0.25° mean": float(year["random_forest"].mean()),
    "Mean absolute difference RF vs bilinear": float(abs(year["random_forest"]
                                                         - year["bilinear"]).mean()),
}).round(2)
print(check)
check.to_csv("results/step6/map_check_2024.csv", header=["µg/m³"])
print("Maps saved in results/step6/")
