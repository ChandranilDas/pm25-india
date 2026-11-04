#!/usr/bin/env python3
"""
Embeds the generated high-resolution diagram outputs into Code.ipynb cells as standard
Jupyter notebook display_data (image/png base64) outputs.
"""

import json
import base64
import os

with open("Code.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def img_output(path, desc="<Figure size 800x600 with 1 Axes>"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [desc]
        },
        "metadata": {}
    }

# 1. Cell 6: EDA Monthly & Mean Maps
eda1 = img_output("results/step2/monthly_pm25_by_region.png", "<Figure size 1000x500 with 1 Axes>")
eda2 = img_output("results/step2/map_mean_pm25.png", "<Figure size 800x600 with 1 Axes>")
if eda1:
    nb["cells"][6]["outputs"].append(eda1)
if eda2:
    nb["cells"][6]["outputs"].append(eda2)

# 2. Cell 10: Spatial Blocking Split Map (Slide 7)
split_img = img_output("results/step3/split_map.png", "<Figure size 600x600 with 1 Axes>")
if split_img:
    nb["cells"][10]["outputs"].append(split_img)

# 3. Cell 18: Feature Importance (Slide 11)
feat_img = img_output("results/step5/feature_importance.png", "<Figure size 700x500 with 1 Axes>")
if feat_img:
    nb["cells"][18]["outputs"].append(feat_img)

# 4. Cell 20: Regional & Seasonal Breakdown & Scatter Plot (Slide 9 & 10)
r2_reg = img_output("results/step5/r2_by_region.png", "<Figure size 800x500 with 1 Axes>")
r2_seas = img_output("results/step5/r2_by_season.png", "<Figure size 800x500 with 1 Axes>")
scatter_img = img_output("results/step5/scatter_random_forest.png", "<Figure size 600x600 with 1 Axes>")
if r2_reg:
    nb["cells"][20]["outputs"].append(r2_reg)
if r2_seas:
    nb["cells"][20]["outputs"].append(r2_seas)
if scatter_img:
    nb["cells"][20]["outputs"].append(scatter_img)

# 5. Cell 22: High-Resolution Spatial Enhancement 3-Panel & 2024 Maps (Slide 12)
map_ex = img_output("results/step6/map_example_day.png", "<Figure size 1200x500 with 3 Axes>")
map_2024 = img_output("results/step6/map_2024_mean.png", "<Figure size 800x600 with 1 Axes>")
if map_ex:
    nb["cells"][22]["outputs"].append(map_ex)
if map_2024:
    nb["cells"][22]["outputs"].append(map_2024)

with open("Code.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Injected live diagram outputs into Code.ipynb successfully!")

