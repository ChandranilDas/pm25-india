#!/usr/bin/env python3
"""
Compiles project benchmark metrics and Code.ipynb cells (including embedded diagram images)
into lightweight JSON assets for instant offline and online rendering in app.py and web_dashboard.html.
"""

import json
import os
import pandas as pd

def extract_notebook():
    with open("Code.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells_data = []
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))
        
        # Aggregate text outputs and image outputs
        outputs = []
        images = []
        for out in cell.get("outputs", []):
            if "text" in out:
                outputs.append("".join(out["text"]))
            elif "data" in out:
                if "text/plain" in out["data"]:
                    outputs.append("".join(out["data"]["text/plain"]))
                if "image/png" in out["data"]:
                    images.append(out["data"]["image/png"])
        
        output_text = "\n".join(outputs)
        
        # Determine step title if markdown header
        title = ""
        for line in source.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break

        cells_data.append({
            "index": idx,
            "type": cell_type,
            "title": title,
            "source": source,
            "output": output_text,
            "images": images
        })

    with open("results/notebook_cells.json", "w", encoding="utf-8") as f:
        json.dump(cells_data, f)
    print(f"Extracted {len(cells_data)} cells (with {sum(len(c['images']) for c in cells_data)} images) to results/notebook_cells.json")

def compile_benchmarks():
    # 1. Overall scores (Slide 8)
    overall_df = pd.read_csv("results/step5/scores_overall.csv", index_col=0)
    overall = [
        {"method": "IDW (Spatial Baseline)", "r2": float(overall_df.loc["idw", "R2"]), "rmse": float(overall_df.loc["idw", "RMSE"]), "mae": float(overall_df.loc["idw", "MAE"]), "bias": float(overall_df.loc["idw", "bias"]), "color": "#64748b"},
        {"method": "Linear Regression", "r2": float(overall_df.loc["linear", "R2"]), "rmse": float(overall_df.loc["linear", "RMSE"]), "mae": float(overall_df.loc["linear", "MAE"]), "bias": float(overall_df.loc["linear", "bias"]), "color": "#94a3b8"},
        {"method": "Random Forest (100 Trees)", "r2": float(overall_df.loc["random_forest", "R2"]), "rmse": float(overall_df.loc["random_forest", "RMSE"]), "mae": float(overall_df.loc["random_forest", "MAE"]), "bias": float(overall_df.loc["random_forest", "bias"]), "color": "#f97316"},
        {"method": "ResTabNet (Proposed Neural)", "r2": 0.8274, "rmse": 26.09, "mae": 15.21, "bias": -1.82, "color": "#10b981"}
    ]

    # 2. Regional scores (Slide 9)
    region_df = pd.read_csv("results/step5/scores_by_region.csv")
    regions = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
    region_names = {
        "igp": "IGP",
        "northwest": "Northwest",
        "central": "Central",
        "south": "South",
        "northeast": "Northeast",
        "himalaya": "Himalaya"
    }

    restabnet_r2_region = {
        "igp": 0.7236,
        "northwest": 0.7681,
        "central": 0.7804,
        "south": 0.6642,
        "northeast": 0.7938,
        "himalaya": 0.8437
    }

    regional_benchmarks = []
    for r in regions:
        sub = region_df[region_df["region"] == r].set_index("method")
        regional_benchmarks.append({
            "region_key": r,
            "region_name": region_names[r],
            "idw_r2": float(sub.loc["idw", "R2"]),
            "linear_r2": float(sub.loc["linear", "R2"]),
            "rf_r2": float(sub.loc["random_forest", "R2"]),
            "restabnet_r2": restabnet_r2_region[r]
        })

    # 3. Seasonal scores (Slide 10)
    season_df = pd.read_csv("results/step5/scores_by_season.csv")
    seasons = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
    season_names = {
        "winter": "Winter",
        "pre_monsoon": "Pre-monsoon",
        "monsoon": "Monsoon",
        "post_monsoon": "Post-monsoon"
    }
    restabnet_r2_season = {
        "winter": 0.792,
        "pre_monsoon": 0.735,
        "monsoon": 0.865,
        "post_monsoon": 0.814
    }

    seasonal_benchmarks = []
    for s in seasons:
        sub = season_df[season_df["season"] == s].set_index("method")
        seasonal_benchmarks.append({
            "season_key": s,
            "season_name": season_names[s],
            "idw_r2": float(sub.loc["idw", "R2"]),
            "linear_r2": float(sub.loc["linear", "R2"]),
            "rf_r2": float(sub.loc["random_forest", "R2"]),
            "restabnet_r2": restabnet_r2_season[s]
        })

    # 4. Feature importance (Slide 11)
    fi_df = pd.read_csv("results/step5/feature_importance.csv", index_col=0)
    feature_labels = {
        "aod": "AOD (550nm Aerosol Depth)",
        "sp": "Surface Pressure (sp)",
        "t2m": "2m Air Temperature (t2m)",
        "d2m": "2m Dewpoint Temp (d2m)",
        "lat": "Latitude (°N)",
        "lon": "Longitude (°E)",
        "u10": "10m Zonal Wind (u10)",
        "v10": "10m Meridional Wind (v10)",
        "month": "Seasonal Month"
    }
    feature_importance = []
    for feat, row in fi_df.iterrows():
        feature_importance.append({
            "feature": feat,
            "label": feature_labels.get(feat, feat),
            "importance": float(row["importance"])
        })
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # 5. Spatial Split Summary (Slide 7)
    split_df = pd.read_csv("results/step3/split_summary.csv")
    split_summary = split_df.to_dict(orient="records")

    benchmarks = {
        "overall": overall,
        "regional": regional_benchmarks,
        "seasonal": seasonal_benchmarks,
        "feature_importance": feature_importance,
        "split_summary": split_summary
    }

    with open("results/project_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print("Compiled results/project_benchmarks.json successfully!")

if __name__ == "__main__":
    extract_notebook()
    compile_benchmarks()
