# Step 5: How good are the predictions at the test cells?
#
# Scores (all in µg/m³ except R²):
#   RMSE  root mean squared error (lower is better)
#   MAE   mean absolute error (lower is better)
#   R²    share of the variation explained (1 is perfect, 0 = no better than the mean)
#   bias  mean of (prediction - CAMS); below 0 means the method predicts too low
# Scores are computed for all test cells, then for each region and each season.
# Then: which predictors matter most to the random forest.
#
# Run: conda run -n pm25 python step5_evaluate.py

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("results/step5", exist_ok=True)
METHODS = ["idw", "linear", "random_forest"]
FEATURES = ["aod", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon", "month"]

test = pd.read_csv("results/step4/test_predictions.csv")


def scores(observed, predicted):
    error = predicted - observed
    return {
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(np.abs(error)),
        "R2": 1 - np.sum(error**2) / np.sum((observed - observed.mean()) ** 2),
        "bias": np.mean(error),
    }


# 1. All test cells together.
overall = pd.DataFrame({m: scores(test["pm25"], test[m]) for m in METHODS}).T
print("All test cells:")
print(overall.round(3))
overall.round(3).to_csv("results/step5/scores_overall.csv")

# 2. By region and by season.
for group in ["region", "season"]:
    rows = []
    for name, part in test.groupby(group):
        for m in METHODS:
            rows.append({group: name, "method": m, **scores(part["pm25"], part[m])})
    result = pd.DataFrame(rows)
    print(f"\nR² by {group}:")
    print(result.pivot(index=group, columns="method", values="R2").round(2))
    result.round(3).to_csv(f"results/step5/scores_by_{group}.csv", index=False)

    result.pivot(index=group, columns="method", values="R2")[METHODS].plot.bar(figsize=(9, 4))
    plt.ylabel("R²")
    plt.title(f"R² at test cells by {group}")
    plt.xticks(rotation=0)
    plt.savefig(f"results/step5/r2_by_{group}.png", dpi=120, bbox_inches="tight")
    plt.close()

# 3. Predicted against CAMS for the random forest.
plt.figure(figsize=(5, 5))
plt.hexbin(test["pm25"], test["random_forest"], gridsize=60, bins="log", cmap="viridis")
plt.plot([0, 500], [0, 500], "r--")
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel("CAMS PM2.5 (µg/m³)")
plt.ylabel("Random forest PM2.5 (µg/m³)")
plt.title("Random forest at test cells")
plt.savefig("results/step5/scatter_random_forest.png", dpi=120, bbox_inches="tight")
plt.close()

# 4. Feature importance of the random forest (how much each predictor is used).
forest = joblib.load("results/step4/random_forest.joblib")
importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values()
print("\nRandom forest feature importance:")
print(importance.round(3))
importance.round(3).to_csv("results/step5/feature_importance.csv", header=["importance"])
importance.plot.barh(figsize=(6, 4))
plt.xlabel("Importance")
plt.title("Random forest feature importance")
plt.savefig("results/step5/feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nTables and figures saved in results/step5/")
