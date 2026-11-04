#!/usr/bin/env python3
"""
Generate publication-grade High-Resolution Spatial Enhancement (0.25°) maps
addressing Research Question 6 (ML downscaling vs standard spatial interpolation).

Generates:
  1. Example severe pollution day (2024-11-15, post-monsoon stubble burning in IGP)
  2. Annual average (2024 full year)
  3. High-frequency spatial residual maps (ML - Bilinear) showing localized terrain/urban enhancements
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = "/tmp/.keras"

import keras

os.makedirs("results/step6", exist_ok=True)
EXAMPLE_DAY = "2024-11-15"

# Load dataset
data = xr.open_dataset("data/pm25_daily.nc")
data = data.rename({"pm2p5": "pm25", "aod550": "aod"})

# Load trained ResTabNet model and normalization parameters
model = keras.models.load_model("pm25_model.h5", compile=False)
norm_stats = np.load("results/step4/norm_params.npz")
norm_mean = norm_stats["mean"]
norm_std = norm_stats["std"]

# High-resolution 0.25 deg grid
fine_lat = np.arange(float(data["lat"].min()), float(data["lat"].max()) + 0.01, 0.25)
fine_lon = np.arange(float(data["lon"].min()), float(data["lon"].max()) + 0.01, 0.25)

# Mask for land points inside India
fine_in_india = (
    data["in_india"].astype(int).interp(lat=fine_lat, lon=fine_lon, method="nearest") == 1
)

def compute_fine_maps(first_day, last_day):
    """Bilinear and ML-predicted PM2.5 on the 0.25° grid."""
    days = data[["aod", "u10", "v10", "t2m", "d2m", "sp", "pm25"]].sel(time=slice(first_day, last_day))
    fine = days.interp(lat=fine_lat, lon=fine_lon) # bilinear interpolation of atmospheric fields
    table = fine.to_dataframe().reset_index()

    # Compute engineered features
    table["month"] = table["time"].dt.month
    u10 = table["u10"].to_numpy()
    v10 = table["v10"].to_numpy()
    t2m = table["t2m"].to_numpy()
    d2m = table["d2m"].to_numpy()
    aod = table["aod"].to_numpy()
    sp = table["sp"].to_numpy()
    lat = table["lat"].to_numpy()
    lon = table["lon"].to_numpy()
    month = table["month"].to_numpy()

    ws = np.sqrt(u10**2 + v10**2)
    rh_proxy = t2m - d2m
    sin_mon = np.sin(2.0 * np.pi * month / 12.0)
    cos_mon = np.cos(2.0 * np.pi * month / 12.0)
    aod_vent = aod / (ws + 0.5)

    X_mat = np.column_stack([
        aod, u10, v10, ws, t2m, d2m, rh_proxy, sp, lat, lon, sin_mon, cos_mon, aod_vent
    ]).astype(np.float32)

    X_scaled = (X_mat - norm_mean) / norm_std
    preds = model.predict(X_scaled, batch_size=4096, verbose=0).flatten()
    table["ml_pm25"] = preds
    table = table.rename(columns={"pm25": "bilinear_pm25"})

    mean_grid = table.groupby(["lat", "lon"])[["bilinear_pm25", "ml_pm25"]].mean().to_xarray()
    mean_grid["residual_diff"] = mean_grid["ml_pm25"] - mean_grid["bilinear_pm25"]
    return mean_grid.where(fine_in_india)


def plot_4panel(cams_coarse, fine_grid, title, out_path, vmax=250):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    
    # 1. Coarse CAMS
    im0 = axes[0].pcolormesh(cams_coarse["lon"], cams_coarse["lat"], cams_coarse, cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[0].set_title("1. CAMS EAC4 (Coarse 0.75° ~80 km)", fontsize=11, fontweight="bold")
    axes[0].set_aspect("equal")

    # 2. Bilinear Interpolation
    im1 = axes[1].pcolormesh(fine_grid["lon"], fine_grid["lat"], fine_grid["bilinear_pm25"], cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[1].set_title("2. Standard Bilinear Interpolation (0.25° ~28 km)", fontsize=11, fontweight="bold")
    axes[1].set_aspect("equal")

    # 3. ML Downscaled (ResTabNet)
    im2 = axes[2].pcolormesh(fine_grid["lon"], fine_grid["lat"], fine_grid["ml_pm25"], cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[2].set_title("3. ML Super-Resolution ResTabNet (0.25° ~28 km)", fontsize=11, fontweight="bold")
    axes[2].set_aspect("equal")

    # 4. High-Frequency Enhancement Residual (ML - Bilinear)
    diff = fine_grid["residual_diff"]
    im3 = axes[3].pcolormesh(diff["lon"], diff["lat"], diff, cmap="coolwarm", vmin=-40, vmax=40)
    axes[3].set_title("4. Detail Injection (ML minus Bilinear)", fontsize=11, fontweight="bold")
    axes[3].set_aspect("equal")

    # Colorbars
    fig.colorbar(im2, ax=axes[:3], shrink=0.8, label="PM2.5 Concentration (µg/m³)")
    fig.colorbar(im3, ax=axes[3], shrink=0.8, label="Difference (µg/m³)")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


print("Generating fine maps for 2024 annual average...")
grid_year = compute_fine_maps("2024-01-01", "2024-12-31")
cams_year = data["pm25"].sel(time="2024").mean("time").where(data["in_india"])
plot_4panel(
    cams_year, grid_year,
    "Spatial Enhancement of PM2.5 over India (2024 Annual Average)",
    "results/step6/map_2024_annual_4panel.png",
    vmax=180
)

print("Generating fine maps for severe episodic day (2024-11-15)...")
grid_day = compute_fine_maps(EXAMPLE_DAY, EXAMPLE_DAY)
cams_day = data["pm25"].sel(time=EXAMPLE_DAY).where(data["in_india"])
plot_4panel(
    cams_day, grid_day,
    f"Spatial Enhancement of PM2.5 over India (Episodic Stubble Burning: {EXAMPLE_DAY})",
    "results/step6/map_example_day_4panel.png",
    vmax=380
)

# Conservation check summary
cams_mean = float(cams_year.mean())
bilinear_mean = float(grid_year["bilinear_pm25"].mean())
ml_mean = float(grid_year["ml_pm25"].mean())
mad = float(abs(grid_year["residual_diff"]).mean())

print("\n--- SPATIAL CONSERVATION SUMMARY (2024 India Mean) ---")
print(f"CAMS Coarse (0.75°):           {cams_mean:.2f} ug/m3")
print(f"Bilinear Interpolation (0.25°): {bilinear_mean:.2f} ug/m3")
print(f"ML ResTabNet (0.25°):          {ml_mean:.2f} ug/m3")
print(f"Mean Absolute Difference:       {mad:.2f} ug/m3")

df_summary = pd.DataFrame([{
    "CAMS_075_mean": cams_mean,
    "Bilinear_025_mean": bilinear_mean,
    "ML_025_mean": ml_mean,
    "Mean_Absolute_Diff": mad
}])
df_summary.to_csv("results/step6/spatial_enhancement_summary.csv", index=False)
print("Saved results/step6/spatial_enhancement_summary.csv")

