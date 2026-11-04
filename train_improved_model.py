#!/usr/bin/env python3
"""
Train the High-Performance, Parameter-Efficient ResTabNet Model
for EE708 Term Project (Project 3).

Features:
  - 9 base CAMS predictors
  - 4 physically-grounded engineered features:
      1. ws: wind speed magnitude = sqrt(u10^2 + v10^2) (ventilation)
      2. rh_proxy: dewpoint depression = t2m - d2m (humidity / fog proxy)
      3. sin_mon, cos_mon: cyclical annual seasonality
      4. aod_vent: stagnation index = aod / (ws + 0.5)
  - Residual Tabular Neural Network (ResTabNet) with skip connections and Huber loss
  - Serialization to pm25_model.h5 (Keras format)
"""

import csv
import math
import os
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = "/tmp/.keras"

import keras
from keras import layers

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

    return [aod, u10, v10, ws, t2m, d2m, rh_proxy, sp, lat, lon, sin_mon, cos_mon, aod_vent]

FEATURE_NAMES = [
    "aod", "u10", "v10", "ws", "t2m", "d2m", "rh_proxy", "sp", "lat", "lon", "sin_mon", "cos_mon", "aod_vent"
]

def main():
    print("=" * 78)
    print(" Training Improved Parameter-Efficient ResTabNet Model for PM2.5 Prediction")
    print("=" * 78)

    # 1. Load test cells
    test_cells = set()
    with open("data/cells.csv", "r") as f:
        for r in csv.DictReader(f):
            if r.get("set") == "test":
                test_cells.add((float(r["lat"]), float(r["lon"])))
    print(f"Loaded {len(test_cells)} held-out spatial test cells.")

    # 2. Ingest dataset
    print("Extracting features from data/pm25_daily.csv...")
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_meta = []

    with open("data/pm25_daily.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = extract_features(row)
            pm25 = float(row["pm25"])
            lat, lon = float(row["lat"]), float(row["lon"])

            if (lat, lon) in test_cells:
                X_test.append(feats)
                y_test.append(pm25)
                test_meta.append((row["date"], lat, lon, row["region"], row["season"]))
            else:
                X_train.append(feats)
                y_train.append(pm25)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Normalization parameters
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    os.makedirs("results/step4", exist_ok=True)
    np.savez("results/step4/norm_params.npz", mean=mean, std=std, features=FEATURE_NAMES)
    print("Saved normalization parameters to results/step4/norm_params.npz")

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # 4. ResTabNet Architecture
    inputs = layers.Input(shape=(len(FEATURE_NAMES),), name="atmospheric_features")
    x0 = layers.Dense(128, activation="relu", name="stem_dense")(inputs)
    x0 = layers.BatchNormalization(name="stem_bn")(x0)

    # Residual Block 1
    x1 = layers.Dense(128, activation="relu", name="res1_dense1")(x0)
    x1 = layers.Dropout(0.1, name="res1_drop")(x1)
    x1 = layers.Dense(128, name="res1_dense2")(x1)
    x1 = layers.add([x0, x1], name="res1_add")
    x1 = layers.Activation("relu", name="res1_act")(x1)
    x1 = layers.BatchNormalization(name="res1_bn")(x1)

    # Residual Block 2
    x2 = layers.Dense(64, activation="relu", name="res2_dense1")(x1)
    x2 = layers.Dropout(0.05, name="res2_drop")(x2)
    x2 = layers.Dense(64, name="res2_dense2")(x2)
    x0_proj = layers.Dense(64, name="proj_dense")(x1)
    x2 = layers.add([x0_proj, x2], name="res2_add")
    x2 = layers.Activation("relu", name="res2_act")(x2)
    x2 = layers.BatchNormalization(name="res2_bn")(x2)

    # Regressor Head
    x3 = layers.Dense(32, activation="relu", name="head_dense")(x2)
    outputs = layers.Dense(1, activation="linear", name="pm25_prediction")(x3)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResTabNet_PM25")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="huber",
        metrics=["mae"]
    )
    model.summary()

    # 5. Training with callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    t0 = time.time()
    history = model.fit(
        X_train_norm, y_train,
        batch_size=1024,
        epochs=10,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Training completed in {time.time() - t0:.2f} seconds.")

    # 6. Save model in .h5 format
    model.save("pm25_model.h5")
    model.save("results/step4/pm25_model.h5")
    print("Model saved to pm25_model.h5 and results/step4/pm25_model.h5")

    # 7. Evaluate on held-out test cells
    preds = model.predict(X_test_norm, batch_size=2048, verbose=0).flatten()
    err = preds - y_test
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((y_test - np.mean(y_test))**2))
    r2 = float(1.0 - np.sum(err**2) / ss_tot)
    bias = float(np.mean(err))

    print("\n" + "=" * 78)
    print(" OVERALL TEST SET EVALUATION (112 UNSEEN GRID CELLS)")
    print("=" * 78)
    print(f"  RMSE: {rmse:8.3f} ug/m3")
    print(f"  MAE:  {mae:8.3f} ug/m3")
    print(f"  R2:   {r2:8.4f}")
    print(f"  Bias: {bias:8.3f} ug/m3")
    print("=" * 78)

    # Save predictions
    with open("results/step4/nn_test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "lat", "lon", "region", "season", "cams_pm25", "restabnet_pm25"])
        for meta, true_v, pred_v in zip(test_meta, y_test, preds):
            writer.writerow([meta[0], meta[1], meta[2], meta[3], meta[4], true_v, pred_v])

    # Regional analysis
    regions = [m[3] for m in test_meta]
    seasons = [m[4] for m in test_meta]
    
    print("\n--- REGIONAL BREAKDOWN ---")
    reg_rows = []
    for reg in sorted(set(regions)):
        idx = [i for i, r in enumerate(regions) if r == reg]
        sub_err = err[idx]
        sub_true = y_test[idx]
        r_rmse = float(np.sqrt(np.mean(sub_err**2)))
        r_mae = float(np.mean(np.abs(sub_err)))
        r_ss_tot = float(np.sum((sub_true - np.mean(sub_true))**2))
        r_r2 = float(1.0 - np.sum(sub_err**2) / r_ss_tot) if r_ss_tot > 0 else 0.0
        r_bias = float(np.mean(sub_err))
        print(f"  {reg:<15} | N={len(idx):<6} | RMSE={r_rmse:<6.2f} | MAE={r_mae:<6.2f} | R2={r_r2:<6.3f} | Bias={r_bias:<6.2f}")
        reg_rows.append({"region": reg, "rmse": r_rmse, "mae": r_mae, "r2": r_r2, "bias": r_bias})

    print("\n--- SEASONAL BREAKDOWN ---")
    seas_rows = []
    for seas in sorted(set(seasons)):
        idx = [i for i, s in enumerate(seasons) if s == seas]
        sub_err = err[idx]
        sub_true = y_test[idx]
        s_rmse = float(np.sqrt(np.mean(sub_err**2)))
        s_mae = float(np.mean(np.abs(sub_err)))
        s_ss_tot = float(np.sum((sub_true - np.mean(sub_true))**2))
        s_r2 = float(1.0 - np.sum(sub_err**2) / s_ss_tot) if s_ss_tot > 0 else 0.0
        s_bias = float(np.mean(sub_err))
        print(f"  {seas:<15} | N={len(idx):<6} | RMSE={s_rmse:<6.2f} | MAE={s_mae:<6.2f} | R2={s_r2:<6.3f} | Bias={s_bias:<6.2f}")
        seas_rows.append({"season": seas, "rmse": s_rmse, "mae": s_mae, "r2": s_r2, "bias": s_bias})

if __name__ == "__main__":
    main()

