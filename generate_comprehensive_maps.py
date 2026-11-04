#!/usr/bin/env python3
"""
Generate comprehensive high-resolution 0.25° spatial downscaling maps
comparing:
  1. CAMS 0.75° Coarse Reanalysis
  2. Standard Bilinear Interpolation 0.25°
  3. Random Forest 0.25°
  4. HistGradientBoosting 0.25°
  5. ResTabNet (Deep Residual Network) 0.25°
along with Spatial Difference / Detail Injection maps (ResTabNet - Bilinear).
"""

import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "torch"
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = "/tmp/.keras"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/.mpl"

import keras
import joblib

os.makedirs("results/step6", exist_ok=True)

# Load data
with h5py.File("data/pm25_daily.nc", "r") as f:
    lats = f["lat"][:]
    lons = f["lon"][:]
    in_india = f["in_india"][:]
    
    # 2024 annual mean (days 1461 to 1827)
    pm25_2024 = f["pm2p5"][1461:].mean(axis=0)
    aod_2024 = f["aod550"][1461:].mean(axis=0)
    u10_2024 = f["u10"][1461:].mean(axis=0)
    v10_2024 = f["v10"][1461:].mean(axis=0)
    t2m_2024 = f["t2m"][1461:].mean(axis=0)
    d2m_2024 = f["d2m"][1461:].mean(axis=0)
    sp_2024 = f["sp"][1461:].mean(axis=0)
    
    # Example severe day: 2024-11-15 (day index ~ 1780)
    # 2020-01-01 to 2024-11-15 is 1780 days
    idx_day = 1780
    pm25_day = f["pm2p5"][idx_day]
    aod_day = f["aod550"][idx_day]
    u10_day = f["u10"][idx_day]
    v10_day = f["v10"][idx_day]
    t2m_day = f["t2m"][idx_day]
    d2m_day = f["d2m"][idx_day]
    sp_day = f["sp"][idx_day]

# Fine 0.25 deg grid
fine_lats = np.arange(lats.min(), lats.max() + 0.01, 0.25)
fine_lons = np.arange(lons.min(), lons.max() + 0.01, 0.25)
mesh_lon, mesh_lat = np.meshgrid(fine_lons, fine_lats)
query_pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

india_interp = RegularGridInterpolator((lats, lons), in_india, method="nearest", bounds_error=False, fill_value=0)
fine_in_india = (india_interp(query_pts) > 0.5).reshape(mesh_lat.shape)

# Load trained models
rf = joblib.load("results/step4/random_forest.joblib")
hgb = joblib.load("results/step4/hgb_model.joblib")
advectsr_data = joblib.load("results/step4/advectsr_model.joblib")
advectsr_mean = advectsr_data["mean_model"]
nn = keras.models.load_model("pm25_model.h5", compile=False)
norm_stats = np.load("results/step4/norm_params.npz")
norm_mean = norm_stats["mean"]
norm_std = norm_stats["std"]

ADVECTSR_FEATURES = [
    "aod", "u10", "v10", "ws", "u_dir", "v_dir", "adv_flux", "aod_vent",
    "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon",
    "wind_himalaya_norm", "wind_himalaya_para"
]


def run_pipeline(pm25_coarse, aod_coarse, u10_coarse, v10_coarse, t2m_coarse, d2m_coarse, sp_coarse, month_val):
    # Bilinear of PM2.5
    pm25_interp = RegularGridInterpolator((lats, lons), pm25_coarse, method="linear", bounds_error=False, fill_value=np.nan)
    bilinear = pm25_interp(query_pts).reshape(mesh_lat.shape)
    
    # Interp features
    iaod = RegularGridInterpolator((lats, lons), aod_coarse)(query_pts)
    iu10 = RegularGridInterpolator((lats, lons), u10_coarse)(query_pts)
    iv10 = RegularGridInterpolator((lats, lons), v10_coarse)(query_pts)
    it2m = RegularGridInterpolator((lats, lons), t2m_coarse)(query_pts)
    id2m = RegularGridInterpolator((lats, lons), d2m_coarse)(query_pts)
    isp  = RegularGridInterpolator((lats, lons), sp_coarse)(query_pts)
    
    # 9 features for RF
    rf_df = pd.DataFrame(np.column_stack([
        iaod, iu10, iv10, it2m, id2m, isp,
        query_pts[:, 0], query_pts[:, 1], np.full(len(query_pts), month_val)
    ]), columns=["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"])
    
    # 13 features for HGB and ResTabNet
    ws = np.sqrt(iu10**2 + iv10**2)
    rh_proxy = it2m - id2m
    sin_mon = np.sin(2.0 * np.pi * month_val / 12.0)
    cos_mon = np.cos(2.0 * np.pi * month_val / 12.0)
    aod_vent = iaod / (ws + 0.5)
    
    hgb_df = pd.DataFrame(np.column_stack([
        iaod, iu10, iv10, ws, it2m, id2m, rh_proxy, isp,
        query_pts[:, 0], query_pts[:, 1],
        np.full(len(query_pts), sin_mon), np.full(len(query_pts), cos_mon), aod_vent
    ]), columns=["aod", "u10", "v10", "ws", "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon", "aod_vent"])
    
    # 18 features for AdvectSR-PM (Physics-Informed ML)
    u_dir = iu10 / (ws + 0.1)
    v_dir = iv10 / (ws + 0.1)
    adv_flux = iaod * ws
    theta = np.deg2rad(30.0)
    wind_norm = -iu10 * np.sin(theta) + iv10 * np.cos(theta)
    wind_para = iu10 * np.cos(theta) + iv10 * np.sin(theta)

    adv_df = pd.DataFrame(np.column_stack([
        iaod, iu10, iv10, ws, u_dir, v_dir, adv_flux, aod_vent,
        it2m, id2m, rh_proxy, isp,
        query_pts[:, 0], query_pts[:, 1],
        np.full(len(query_pts), sin_mon), np.full(len(query_pts), cos_mon),
        wind_norm, wind_para
    ]), columns=ADVECTSR_FEATURES)

    # Predictions
    rf_pred = rf.predict(rf_df).reshape(mesh_lat.shape)
    hgb_pred = hgb.predict(hgb_df).reshape(mesh_lat.shape)
    adv_pred = np.clip(advectsr_mean.predict(adv_df), 0, None).reshape(mesh_lat.shape)
    
    nn_scaled = (hgb_df.values - norm_mean) / norm_std
    nn_pred = nn.predict(nn_scaled, batch_size=4096, verbose=0).flatten().reshape(mesh_lat.shape)
    
    # Mask outside India
    coarse_masked = np.where(in_india > 0.5, pm25_coarse, np.nan)
    bilinear_masked = np.where(fine_in_india, bilinear, np.nan)
    rf_masked = np.where(fine_in_india, rf_pred, np.nan)
    hgb_masked = np.where(fine_in_india, hgb_pred, np.nan)
    adv_masked = np.where(fine_in_india, adv_pred, np.nan)
    nn_masked = np.where(fine_in_india, nn_pred, np.nan)
    
    diff_res = np.where(fine_in_india, nn_pred - bilinear, np.nan)
    diff_adv = np.where(fine_in_india, adv_pred - bilinear, np.nan)
    diff_hgb = np.where(fine_in_india, hgb_pred - bilinear, np.nan)
    
    return {
        "coarse": coarse_masked,
        "bilinear": bilinear_masked,
        "rf": rf_masked,
        "hgb": hgb_masked,
        "advectsr": adv_masked,
        "nn": nn_masked,
        "diff_res": diff_res,
        "diff_adv": diff_adv,
        "diff_hgb": diff_hgb
    }


def plot_6panel(maps, title, out_path, vmax=200):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5.5), constrained_layout=True)
    
    panels = [
        ("CAMS 0.75°\n(Coarse Reanalysis)", lons, lats, maps["coarse"]),
        ("Bilinear Interp 0.25°\n(Standard Baseline)", fine_lons, fine_lats, maps["bilinear"]),
        ("Random Forest 0.25°\n(R² = 0.742, 166 MB)", fine_lons, fine_lats, maps["rf"]),
        ("HistGradientBoost 0.25°\n(R² = 0.754, 340 KB)", fine_lons, fine_lats, maps["hgb"]),
        ("AdvectSR-PM (PIML) 0.25°\n(R² = 0.777, CCE=0.0)", fine_lons, fine_lats, maps["advectsr"]),
        ("ResTabNet (Deep Res) 0.25°\n(R² = 0.827, 773 KB)", fine_lons, fine_lats, maps["nn"])
    ]
    
    for ax, (name, x, y, z) in zip(axes, panels):
        im = ax.pcolormesh(x, y, z, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(name, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(68, 98)
        ax.set_ylim(6, 38)
        ax.set_xlabel("Longitude (°E)", fontsize=8.5)
        ax.set_ylabel("Latitude (°N)", fontsize=8.5)
        ax.grid(color="gray", linestyle=":", alpha=0.3)
    
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.35, pad=0.08, aspect=30)
    cb.set_label("Surface PM2.5 Concentration (µg/m³)", fontsize=11, fontweight="bold")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_residual_detail(maps, title, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), constrained_layout=True)
    
    panels = [
        ("ResTabNet minus Bilinear\n(Deep Residual Detail)", maps["diff_res"]),
        ("AdvectSR-PM minus Bilinear\n(Physics Advection Transport Detail)", maps["diff_adv"]),
        ("HistGradientBoosting minus Bilinear\n(Tree Histogram Detail)", maps["diff_hgb"]),
        ("Random Forest minus Bilinear\n(Traditional Forest Detail)", maps["rf"] - maps["bilinear"])
    ]
    
    for ax, (name, z) in zip(axes, panels):
        im = ax.pcolormesh(fine_lons, fine_lats, z, cmap="coolwarm", vmin=-35, vmax=35, shading="auto")
        ax.set_title(name, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(68, 98)
        ax.set_ylim(6, 38)
        ax.set_xlabel("Longitude (°E)", fontsize=8.5)
        ax.set_ylabel("Latitude (°N)", fontsize=8.5)
        ax.grid(color="gray", linestyle=":", alpha=0.3)
    
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.35, pad=0.08, aspect=30)
    cb.set_label("High-Frequency Detail Delta (µg/m³)", fontsize=11, fontweight="bold")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


print("Generating 2024 annual average spatial downscaling...")
maps_2024 = run_pipeline(pm25_2024, aod_2024, u10_2024, v10_2024, t2m_2024, d2m_2024, sp_2024, month_val=6.5)
plot_6panel(maps_2024, "Spatial Downscaling of Surface PM2.5 over India (2024 Annual Average)", "results/step6/map_2024_mean.png", vmax=200)
plot_6panel(maps_2024, "Spatial Downscaling of Surface PM2.5 over India (2024 Annual Average)", "results/step6/map_5panel_comparison_2024.png", vmax=200)
plot_residual_detail(maps_2024, "High-Frequency Spatial Detail Injection vs Standard Bilinear (2024 Mean)", "results/step6/map_enhancement_diff.png")

print("Generating single severe episodic day (2024-11-15 post-monsoon stubble burning)...")
maps_day = run_pipeline(pm25_day, aod_day, u10_day, v10_day, t2m_day, d2m_day, sp_day, month_val=11.0)
plot_6panel(maps_day, "Spatial Downscaling of Surface PM2.5: Severe Episode (2024-11-15 Stubble Burning)", "results/step6/map_example_day.png", vmax=420)
plot_6panel(maps_day, "Spatial Downscaling of Surface PM2.5: Severe Episode (2024-11-15 Stubble Burning)", "results/step6/map_5panel_comparison_example_day.png", vmax=420)

print("\nSpatial enhancement mapping successfully generated across all models!")

print("\nSpatial enhancement mapping successfully generated across all models!")

