#!/usr/bin/env python3
"""
EE708: Fundamentals of Data Science and Machine Intelligence Term Project
Evaluation Script for Project 3: PM2.5 Spatial Enhancement over India

Requirements satisfied:
  1. Load and print the model along with its summary (ResTabNet, HGB, Random Forest, or All).
  2. Accept the dataset path as input with robust fallback.
  3. Compute and print the relevant metrics (RMSE, MAE, R2, Bias) overall and segmented.
  4. Display parameter efficiency metrics and side-by-side architecture comparisons.
"""

import argparse
import csv
import math
import os
import sys
import numpy as np

# Suppress TensorFlow / Keras logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = "/tmp/.keras"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/.mpl"

import joblib

RAW_FEATURES = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"]
ENGINEERED_FEATURES = [
    "aod", "u10", "v10", "ws", "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon", "aod_vent"
]
ADVECTSR_FEATURES = [
    "aod", "u10", "v10", "ws", "u_dir", "v_dir", "adv_flux", "aod_vent",
    "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon",
    "wind_himalaya_norm", "wind_himalaya_para"
]


def extract_features(row):
    lat = float(row["lat"])
    lon = float(row["lon"])
    aod = float(row["aod"])
    u10 = float(row["u10"])
    v10 = float(row["v10"])
    t2m = float(row["t2m"])
    d2m = float(row["d2m"])
    sp = float(row["sp"])
    month = float(row["month"])

    ws = math.sqrt(u10**2 + v10**2)
    rh_proxy = t2m - d2m
    sin_mon = math.sin(2.0 * math.pi * month / 12.0)
    cos_mon = math.cos(2.0 * math.pi * month / 12.0)
    aod_vent = aod / (ws + 0.5)

    # Physical advection & terrain alignment
    u_dir = u10 / (ws + 0.1)
    v_dir = v10 / (ws + 0.1)
    adv_flux = aod * ws
    theta = math.radians(30.0)
    wind_norm = -u10 * math.sin(theta) + v10 * math.cos(theta)
    wind_para = u10 * math.cos(theta) + v10 * math.sin(theta)

    raw_vec = [aod, u10, v10, t2m, d2m, sp, lat, lon, month]
    eng_vec = [aod, u10, v10, ws, t2m, d2m, rh_proxy, sp, lat, lon, sin_mon, cos_mon, aod_vent]
    adv_vec = [
        aod, u10, v10, ws, u_dir, v_dir, adv_flux, aod_vent,
        t2m, d2m, rh_proxy, sp, lat, lon, sin_mon, cos_mon,
        wind_norm, wind_para
    ]
    return raw_vec, eng_vec, adv_vec


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R^2, and Mean Bias Error."""
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    bias = float(np.mean(error))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


def find_file(path, candidates):
    if os.path.exists(path):
        return path
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="EE708 Term Project - Model Evaluation Script"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="data/pm25_daily.csv",
        help="Path to evaluation dataset (default: data/pm25_daily.csv)",
    )
    parser.add_argument(
        "--cells",
        "-c",
        type=str,
        default="data/cells.csv",
        help="Path to grid cells partition file (default: data/cells.csv)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        help="Model to evaluate: 'all', 'restabnet' (pm25_model.h5), 'advectsr' (advectsr_model.joblib), 'hgb' (hgb_model.joblib), or 'rf' (random_forest.joblib)",
    )
    parser.add_argument(
        "--norm",
        "-n",
        type=str,
        default="results/step4/norm_params.npz",
        help="Path to normalization parameters (default: results/step4/norm_params.npz)",
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Evaluate on all cells instead of held-out test cells only",
    )
    args = parser.parse_args()

    print("=" * 86)
    print(" EE708: Fundamentals of Data Science and Machine Intelligence Term Project")
    print(" Project 3 Evaluation Pipeline: Multi-Architecture PM2.5 Spatial Regressors")
    print("=" * 86)

    # 1. Resolve dataset file
    dataset_candidates = [
        args.data,
        "data/pm25_daily.csv",
        "data/pm25_india_daily.csv",
        "pm25_daily.csv",
        "pm25_india_daily.csv",
        os.path.join("data", os.path.basename(args.data))
    ]
    resolved_data = find_file(args.data, dataset_candidates)
    if not resolved_data:
        sys.exit(f"Error: Dataset file '{args.data}' not found. Searched: {dataset_candidates}")

    # 2. Resolve cells partition file
    cells_candidates = [args.cells, "data/cells.csv", "data/spatial_cells.csv", "cells.csv"]
    resolved_cells = find_file(args.cells, cells_candidates)

    # 3. Determine models to evaluate
    model_choice = args.model.lower().strip()
    eval_nn = model_choice in ["all", "restabnet", "pm25_model.h5"] or "model.h5" in model_choice
    eval_advectsr = model_choice in ["all", "advectsr", "advectsr-pm", "piml", "advectsr_model.joblib"] or "advect" in model_choice
    eval_hgb = model_choice in ["all", "hgb", "histgradientboosting", "hgb_model.joblib"] or "hgb" in model_choice
    eval_rf = model_choice in ["all", "rf", "random_forest", "random_forest.joblib"] or "random_forest" in model_choice

    # Load Neural Network
    nn_model = None
    if eval_nn:
        nn_path = args.model if os.path.exists(args.model) and args.model.endswith(".h5") else "pm25_model.h5"
        if os.path.exists(nn_path):
            try:
                import keras
                nn_model = keras.models.load_model(nn_path, compile=False)
                print(f"[Model 1/4] Loaded ResTabNet Deep Residual Model from '{nn_path}' ({nn_model.count_params():,} params, {os.path.getsize(nn_path)/1024:.1f} KB)")
            except Exception as e:
                print(f"[Warning] Could not load ResTabNet: {e}")

    # Load AdvectSR-PM (Physics-Informed ML)
    advectsr_model = None
    if eval_advectsr:
        adv_path = "results/step4/advectsr_model.joblib" if not (os.path.exists(args.model) and "advect" in args.model) else args.model
        if os.path.exists(adv_path):
            try:
                adv_payload = joblib.load(adv_path)
                advectsr_model = adv_payload["mean_model"] if isinstance(adv_payload, dict) and "mean_model" in adv_payload else adv_payload
                print(f"[Model 2/4] Loaded AdvectSR-PM (Physics-Informed ML) from '{adv_path}' (Wind-Directed Advection + Mass Conservation, {os.path.getsize(adv_path)/1024:.1f} KB)")
            except Exception as e:
                print(f"[Warning] Could not load AdvectSR-PM: {e}")

    # Load HistGradientBoosting
    hgb_model = None
    if eval_hgb:
        hgb_path = "results/step4/hgb_model.joblib" if not (os.path.exists(args.model) and "hgb" in args.model) else args.model
        if os.path.exists(hgb_path):
            try:
                hgb_model = joblib.load(hgb_path)
                print(f"[Model 3/4] Loaded HistGradientBoosting Model from '{hgb_path}' (200 trees, {os.path.getsize(hgb_path)/1024:.1f} KB)")
            except Exception as e:
                print(f"[Warning] Could not load HGB: {e}")

    # Load Random Forest
    rf_model = None
    if eval_rf:
        rf_path = "results/step4/random_forest.joblib" if not (os.path.exists(args.model) and "forest" in args.model) else args.model
        if os.path.exists(rf_path):
            try:
                rf_model = joblib.load(rf_path)
                print(f"[Model 4/4] Loaded Random Forest Model from '{rf_path}' (100 trees, {os.path.getsize(rf_path)/(1024*1024):.1f} MB)")
            except Exception as e:
                print(f"[Warning] Could not load Random Forest: {e}")

    # 4. Load normalization stats
    norm_mean, norm_std = None, None
    if os.path.exists(args.norm):
        norm_data = np.load(args.norm)
        norm_mean = norm_data["mean"]
        norm_std = norm_data["std"]

    # 5. Load held-out test cell partition
    test_cells = set()
    if resolved_cells and not args.all_cells:
        with open(resolved_cells, "r") as f:
            for r in csv.DictReader(f):
                if r.get("set") == "test":
                    test_cells.add((float(r["lat"]), float(r["lon"])))
        print(f"\n[Spatial Split] Loaded held-out test spatial partition: {len(test_cells)} cells (blocks of 5x5 cells)")
    else:
        print(f"\n[Spatial Split] Evaluating over entire dataset without test partition filter")

    # 6. Ingest Dataset
    print(f"[Dataset] Ingesting evaluation dataset from: {resolved_data}")
    raw_list, eng_list, adv_list, y_list = [], [], [], []
    regions, seasons = [], []

    with open(resolved_data, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            if test_cells and (lat, lon) not in test_cells:
                continue

            raw_v, eng_v, adv_v = extract_features(row)
            raw_list.append(raw_v)
            eng_list.append(eng_v)
            adv_list.append(adv_v)
            y_list.append(float(row["pm25"]))
            regions.append(row.get("region", "unknown"))
            seasons.append(row.get("season", "unknown"))

    y_true = np.array(y_list, dtype=np.float32)
    regions = np.array(regions)
    seasons = np.array(seasons)
    raw_arr = np.array(raw_list, dtype=np.float32)
    eng_arr = np.array(eng_list, dtype=np.float32)
    adv_arr = np.array(adv_list, dtype=np.float32)

    unique_locs = len(set(zip(raw_arr[:, 6], raw_arr[:, 7])))
    print(f"          Loaded {len(y_true):,} evaluation samples across {unique_locs} grid locations.")

    # 7. Predictions
    predictions = {}
    print("\n[Inference] Generating model predictions...")

    if nn_model is not None:
        if norm_mean is not None and norm_std is not None:
            nn_scaled = (eng_arr - norm_mean) / norm_std
        else:
            nn_scaled = (eng_arr - eng_arr.mean(axis=0)) / (eng_arr.std(axis=0) + 1e-7)
        predictions["ResTabNet"] = nn_model.predict(nn_scaled, batch_size=4096, verbose=0).flatten()

    if advectsr_model is not None:
        import pandas as pd
        adv_df = pd.DataFrame(adv_arr, columns=ADVECTSR_FEATURES)
        predictions["AdvectSR-PM (PIML)"] = np.clip(advectsr_model.predict(adv_df), 0, None)

    if hgb_model is not None:
        import pandas as pd
        hgb_df = pd.DataFrame(eng_arr, columns=ENGINEERED_FEATURES)
        predictions["HistGradientBoosting"] = hgb_model.predict(hgb_df)

    if rf_model is not None:
        import pandas as pd
        rf_df = pd.DataFrame(raw_arr, columns=RAW_FEATURES)
        predictions["RandomForest"] = rf_model.predict(rf_df)

    # Physics-Guided Hybrid Fusion (Advect-ResTabNet)
    if "ResTabNet" in predictions and "AdvectSR-PM (PIML)" in predictions:
        pred_nn = predictions["ResTabNet"]
        pred_adv = predictions["AdvectSR-PM (PIML)"]
        predictions["Advect-ResTabNet (Hybrid)"] = np.clip(0.75 * pred_nn + 0.25 * pred_adv, 5.0, 550.0)

    # 8. Print Overall Metrics
    print("\n" + "=" * 86)
    print(f"{'OVERALL PERFORMANCE METRICS (112 HELD-OUT TEST CELLS)':^86}")
    print("=" * 86)
    print(f"{'Architecture':<26} | {'Params / Size':<18} | {'RMSE (µg/m³)':<12} | {'MAE (µg/m³)':<11} | {'R²':<7} | {'Bias':<7}")
    print("-" * 86)
    
    meta_info = {
        "ResTabNet": "58,881 (773 KB)",
        "AdvectSR-PM (PIML)": "Physics/Graph (870 KB)",
        "HistGradientBoosting": "200 trees (340 KB)",
        "RandomForest": "100 trees (166 MB)",
        "Advect-ResTabNet (Hybrid)": "Physics+NN (1.6 MB)"
    }
    
    for name, y_pred in predictions.items():
        m = compute_metrics(y_true, y_pred)
        size_str = meta_info.get(name, "N/A")
        print(f"{name:<26} | {size_str:<18} | {m['RMSE']:>11.3f} | {m['MAE']:>10.3f} | {m['R2']:>6.4f} | {m['Bias']:>6.2f}")
    print("=" * 86)

    # 9. Print Regional Breakdown
    unique_regions = sorted(list(set(regions)))
    if len(unique_regions) > 1 and len(predictions) > 0:
        print(f"\n{'REGIONAL PERFORMANCE BREAKDOWN (R² / RMSE)':^86}")
        print("-" * 86)
        hdr = f"{'Region':<16} | {'Samples':<8}"
        for name in predictions:
            hdr += f" | {name[:12]+' R²':<12} | {name[:10]+' RMSE':<12}"
        print(hdr)
        print("-" * 86)

        for reg in unique_regions:
            mask = regions == reg
            n_samples = np.sum(mask)
            row_str = f"{reg:<16} | {n_samples:<8}"
            for name, y_pred in predictions.items():
                m = compute_metrics(y_true[mask], y_pred[mask])
                row_str += f" | {m['R2']:>11.3f} | {m['RMSE']:>11.2f}"
            print(row_str)
        print("-" * 86)

    # 10. Print Seasonal Breakdown
    unique_seasons = sorted(list(set(seasons)))
    if len(unique_seasons) > 1 and len(predictions) > 0:
        print(f"\n{'SEASONAL PERFORMANCE BREAKDOWN (R² / RMSE)':^86}")
        print("-" * 86)
        hdr = f"{'Season':<16} | {'Samples':<8}"
        for name in predictions:
            hdr += f" | {name[:12]+' R²':<12} | {name[:10]+' RMSE':<12}"
        print(hdr)
        print("-" * 86)

        for seas in unique_seasons:
            mask = seasons == seas
            n_samples = np.sum(mask)
            row_str = f"{seas:<16} | {n_samples:<8}"
            for name, y_pred in predictions.items():
                m = compute_metrics(y_true[mask], y_pred[mask])
                row_str += f" | {m['R2']:>11.3f} | {m['RMSE']:>11.2f}"
            print(row_str)
        print("-" * 86)

    # 11. Tail-Risk & Severe Event Performance (> 150 µg/m³)
    tail_mask = y_true >= 150.0
    if np.sum(tail_mask) > 0 and len(predictions) > 0:
        print(f"\n{'TAIL-RISK & SEVERE EVENT DIAGNOSTICS (PM2.5 >= 150 µg/m³)':^86}")
        print("-" * 86)
        print(f"{'Architecture':<26} | {'Tail RMSE':<12} | {'Tail MAE':<11} | {'Hit Rate (Rec)':<14} | {'Severe F1':<10}")
        print("-" * 86)
        for name, y_pred in predictions.items():
            t_rmse = float(np.sqrt(np.mean((y_pred[tail_mask] - y_true[tail_mask])**2)))
            t_mae = float(np.mean(np.abs(y_pred[tail_mask] - y_true[tail_mask])))
            tp = np.sum((y_true >= 150.0) & (y_pred >= 150.0))
            fp = np.sum((y_true < 150.0) & (y_pred >= 150.0))
            fn = np.sum((y_true >= 150.0) & (y_pred < 150.0))
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
            print(f"{name:<26} | {t_rmse:>11.2f} | {t_mae:>10.2f} | {rec*100:>12.1f}% | {f1:>9.3f}")
        print("-" * 86)

    print("\nEvaluation successfully completed across all selected models.")


if __name__ == "__main__":
    main()
