# Step 4: Train the two ML models and make the interpolation baseline
#
# Methods, all predicting daily PM2.5 at the test cells:
#   1. IDW (inverse distance weighting): a standard interpolation method. It uses
#      only the PM2.5 of the 8 nearest training cells, closer cells count more.
#   2. Linear regression (ML model 1)
#   3. Random forest (ML model 2)
# The two ML models use the CAMS predictors at the test cell itself.
#
# Run: conda run -n pm25 python step4_train_models.py   (a few minutes)
# Output: results/step4/test_predictions.csv, results/step4/random_forest.joblib

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

os.makedirs("results/step4", exist_ok=True)
FEATURES = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"]

table = pd.read_csv("data/pm25_daily.csv", parse_dates=["date"])
cells = pd.read_csv("data/cells.csv")
table = table.merge(cells[["lat", "lon", "set"]], on=["lat", "lon"])
train = table[table["set"] == "train"]
test = table[table["set"] == "test"].copy()
print("Training rows:", len(train), " Test rows:", len(test))

# 1. IDW. PM2.5 as a table: one row per day, one column per cell.
pm25 = table.pivot_table(index="date", columns=["lat", "lon"], values="pm25")
train_cells = cells[cells["set"] == "train"]
test_cells = cells[cells["set"] == "test"]
idw = {}
for _, cell in test_cells.iterrows():
    # Distance in degrees (simple, good enough to find the nearest cells).
    distance = np.sqrt((train_cells["lat"] - cell["lat"]) ** 2
                       + (train_cells["lon"] - cell["lon"]) ** 2)
    nearest = distance.nsmallest(8)
    weights = 1 / nearest**2
    weights = weights / weights.sum()
    neighbours = train_cells.loc[nearest.index]
    values = pm25[list(zip(neighbours["lat"], neighbours["lon"]))]
    idw[(cell["lat"], cell["lon"])] = values.to_numpy() @ weights.to_numpy()
idw = pd.DataFrame(idw, index=pm25.index)
idw.columns.names = ["lat", "lon"]
idw = idw.stack(["lat", "lon"], future_stack=True).rename("idw").reset_index()
test = test.merge(idw, on=["date", "lat", "lon"])

# 2. Linear regression.
linear = LinearRegression()
linear.fit(train[FEATURES], train["pm25"])
test["linear"] = linear.predict(test[FEATURES])
print("Linear regression coefficients:")
for name, coef in zip(FEATURES, linear.coef_):
    print(f"  {name:6s} {coef:8.3f}")

# 3. Random forest.
forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=10,
                               n_jobs=-1, random_state=42)
forest.fit(train[FEATURES], train["pm25"])
test["random_forest"] = forest.predict(test[FEATURES])
joblib.dump(forest, "results/step4/random_forest.joblib", compress=3)

columns = ["date", "lat", "lon", "region", "season", "pm25", "idw", "linear", "random_forest"]
test[columns].to_csv("results/step4/test_predictions.csv", index=False)
print("Saved results/step4/test_predictions.csv and the random forest model")
