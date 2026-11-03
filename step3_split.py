# Step 3: Choose training cells and test cells
#
# The models must be tested at grid cells they have never seen (project brief).
# Neighbouring cells have almost the same PM2.5, so we do not pick single test
# cells at random. Instead we cut the map into blocks of 5 x 5 grid cells
# (about 380 km) and keep every 5th block, in a diagonal pattern, for testing.
# This spreads the test blocks over the whole country.
#
# Run: conda run -n pm25 python step3_split.py
# Output: data/cells.csv (one row per grid cell: lat, lon, region, set)

import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/step3", exist_ok=True)
table = pd.read_csv("data/pm25_daily.csv")
cells = table[["lat", "lon", "region"]].drop_duplicates().reset_index(drop=True)

# Block row and block column of each cell (the grid starts at 6.0 N, 68.25 E,
# and one cell is 0.75 degrees).
cells["block_row"] = ((cells["lat"] - 6.0) / 0.75).round().astype(int) // 5
cells["block_col"] = ((cells["lon"] - 68.25) / 0.75).round().astype(int) // 5

# Diagonal pattern: a block is a test block when (row + 2 * column) is a multiple of 5.
is_test = (cells["block_row"] + 2 * cells["block_col"]) % 5 == 0
cells["set"] = "train"
cells.loc[is_test, "set"] = "test"

summary = pd.crosstab(cells["region"], cells["set"])
summary["test_share"] = (summary["test"] / (summary["test"] + summary["train"])).round(2)
print(summary)
print("Total test share:", round(is_test.mean(), 2))
summary.to_csv("results/step3/split_summary.csv")
cells.to_csv("data/cells.csv", index=False)

# Map of training and test cells.
plt.figure(figsize=(7, 7))
for name, colour in [("train", "lightgrey"), ("test", "tab:red")]:
    part = cells[cells["set"] == name]
    plt.scatter(part["lon"], part["lat"], c=colour, marker="s", s=60, label=name)
plt.legend()
plt.title("Training and test grid cells")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("results/step3/split_map.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved data/cells.csv and results/step3/split_map.png")
