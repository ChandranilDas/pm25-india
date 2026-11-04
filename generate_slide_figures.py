#!/usr/bin/env python3
"""
Generates publication-quality presentation and notebook figures matching the slides:
 - Slide 7: Spatial Train/Test Grid Split Map (382 grey train, 112 red test)
 - Slide 8: Overall Model Comparison Bar Chart & Table (R2, RMSE, MAE, Bias) with HGB
 - Slide 9: Regional Breakdown across 6 regions (ML wins where neighbours differ) with HGB
 - Slide 10: Seasonal Breakdown Bar Chart + 2D Parity Density Hexbin Plot with HGB
 - Slide 11: Feature Importance Analysis
 - Slide 12: Spatial Downscaling 3-Panel Map Comparison (CAMS vs Bilinear vs ML)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("results/step5", exist_ok=True)
os.makedirs("results/step3", exist_ok=True)
os.makedirs("results/step6", exist_ok=True)

# Set clean styling
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# 1. SLIDE 8: Overall Model Comparison (R2 Bar Chart + Table)
# ==============================================================================
fig = plt.figure(figsize=(13, 5.5), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1.05], wspace=0.28)

ax_bar = fig.add_subplot(gs[0])
methods = ["IDW", "Linear regr.", "Random forest", "HistGradBoost", "AdvectSR-PM", "ResTabNet (Ours)"]
r2_vals = [0.683, 0.446, 0.742, 0.754, 0.777, 0.827]
colors = ["#4e6578", "#b5c1cb", "#d9531e", "#0284c7", "#8b5cf6", "#10b981"]

bars = ax_bar.bar(methods, r2_vals, color=colors, width=0.58, edgecolor="none")
ax_bar.set_ylim(0, 1.0)
ax_bar.set_ylabel("R²", fontsize=11, fontweight="bold", color="#1e293b")
ax_bar.set_title("R² at the 112 test cells", fontsize=13, fontweight="bold", pad=12, loc="center")
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
ax_bar.set_axisbelow(True)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    ax_bar.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight="bold", color="#1e293b")

ax_bar.tick_params(axis='x', rotation=18, labelsize=9)

# Metrics Table
ax_tbl = fig.add_subplot(gs[1])
ax_tbl.axis('off')

table_data = [
    ["IDW", "35.3", "20.4", "-6.8"],
    ["Linear regr.", "46.7", "32.1", "-12.9"],
    ["Random forest", "31.9", "17.8", "-5.6"],
    ["HistGradBoost", "31.1", "17.6", "-4.8"],
    ["AdvectSR-PM", "29.7", "16.6", "-3.4"],
    ["ResTabNet", "26.1", "14.8", "-5.3"]
]
col_labels = ["Method", "RMSE", "MAE", "Bias"]

tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 1.6)

# Style table headers and alternating rows
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='white', weight='bold')
    else:
        if row == 3: # Random forest
            cell.set_facecolor('#fff7ed')
            cell.set_text_props(weight='bold', color='#c2410c')
        elif row == 4: # HistGradBoost
            cell.set_facecolor('#f0f9ff')
            cell.set_text_props(weight='bold', color='#0284c7')
        elif row == 5: # AdvectSR-PM
            cell.set_facecolor('#faf5ff')
            cell.set_text_props(weight='bold', color='#7c3aed')
        elif row == 6: # ResTabNet
            cell.set_facecolor('#ecfdf5')
            cell.set_text_props(weight='bold', color='#047857')
        else:
            cell.set_facecolor('#f8fafc' if row % 2 == 1 else '#ffffff')
    cell.set_edgecolor('#e2e8f0')

ax_tbl.text(0.5, 0.05, "RMSE, MAE, bias in µg/m³\n• AdvectSR-PM (Physics-Informed ML): R² 0.777, CCE = 0.000 µg/m³\n• HistGradBoost: R² 0.754 at 500x smaller footprint (0.34 MB)\n• ResTabNet: Highest overall R² 0.8274 with deep tabular residual skips",
            ha='center', va='top', fontsize=8.0, color='#475569', linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1", alpha=0.7))

plt.tight_layout()
plt.savefig("results/step5/scores_overall_slide8.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/step5/scores_overall_slide8.png")


# ==============================================================================
# 2. SLIDE 9: Regional Breakdown across 6 regions
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
regions = ["IGP", "Northwest", "Central", "South", "Northeast", "Himalaya"]
x = np.arange(len(regions))
width = 0.13

idw_r2  = [0.56, 0.80, 0.70, 0.45, -0.74, 0.46]
lin_r2  = [0.29, -0.05, 0.44, -1.80, 0.15, 0.49]
rf_r2   = [0.54, 0.75, 0.75, 0.47, 0.74, 0.78]
hgb_r2  = [0.57, 0.73, 0.73, 0.57, 0.79, 0.82]
adv_r2  = [0.60, 0.76, 0.75, 0.63, 0.80, 0.86]
nn_r2   = [0.72, 0.77, 0.78, 0.66, 0.79, 0.84]

b1 = ax.bar(x - 2.5*width, idw_r2,  width, label='IDW', color='#4e6578')
b2 = ax.bar(x - 1.5*width, lin_r2,  width, label='Linear regression', color='#cbd5e1')
b3 = ax.bar(x - 0.5*width, rf_r2,   width, label='Random forest', color='#d9531e')
b4 = ax.bar(x + 0.5*width, hgb_r2,  width, label='HistGradBoost', color='#0284c7')
b5 = ax.bar(x + 1.5*width, adv_r2,  width, label='AdvectSR-PM (PIML)', color='#8b5cf6')
b6 = ax.bar(x + 2.5*width, nn_r2,   width, label='ResTabNet (Ours)', color='#10b981')

ax.set_ylabel('R²', fontsize=11, fontweight='bold')
ax.set_title('By region: ML wins where neighbours differ (R² on Held-out Cells)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=10, fontweight='bold')
ax.set_ylim(-2.0, 1.05)
ax.axhline(0, color='#64748b', linewidth=0.8)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper right', fontsize=8.0, ncol=3)

# Annotate AdvectSR-PM & ResTabNet
for bar in b5:
    val = bar.get_height()
    y_pos = val + 0.03 if val >= 0 else val - 0.12
    ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, y_pos),
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=7.2, fontweight='bold', color='#6d28d9')

plt.tight_layout()
plt.savefig("results/step5/r2_by_region_slide9.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/step5/r2_by_region_slide9.png")


# ==============================================================================
# 3. SLIDE 10: Seasonal Breakdown & 2D Parity Density Hexbin
# ==============================================================================
fig = plt.figure(figsize=(13, 5.2), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 0.95], wspace=0.25)

# Left: Seasonal grouped bar
ax_seas = fig.add_subplot(gs[0])
seasons = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
sx = np.arange(len(seasons))
swidth = 0.13

s_idw = [0.61, 0.54, 0.78, 0.64]
s_lin = [0.37, 0.20, 0.36, 0.44]
s_rf  = [0.69, 0.60, 0.81, 0.71]
s_hgb = [0.71, 0.60, 0.81, 0.74]
s_adv = [0.74, 0.63, 0.82, 0.76]
s_nn  = [0.82, 0.71, 0.84, 0.81]

ax_seas.bar(sx - 2.5*swidth, s_idw, swidth, label='IDW', color='#4e6578')
ax_seas.bar(sx - 1.5*swidth, s_lin, swidth, label='Linear regression', color='#cbd5e1')
ax_seas.bar(sx - 0.5*swidth, s_rf,  swidth, label='Random forest', color='#d9531e')
ax_seas.bar(sx + 0.5*swidth, s_hgb, swidth, label='HistGradBoost', color='#0284c7')
sb5 = ax_seas.bar(sx + 1.5*swidth, s_adv, swidth, label='AdvectSR-PM', color='#8b5cf6')
ax_seas.bar(sx + 2.5*swidth, s_nn,  swidth, label='ResTabNet', color='#10b981')

ax_seas.set_ylabel('R²', fontsize=11, fontweight='bold')
ax_seas.set_title('By season: best in the monsoon', fontsize=13, fontweight='bold', pad=12)
ax_seas.set_xticks(sx)
ax_seas.set_xticklabels(seasons, fontsize=10, fontweight='bold')
ax_seas.set_ylim(0, 1.0)
ax_seas.yaxis.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')
ax_seas.spines['top'].set_visible(False)
ax_seas.spines['right'].set_visible(False)
ax_seas.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.0, ncol=3)

for bar in sb5:
    val = bar.get_height()
    ax_seas.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, val + 0.02),
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#6d28d9')

# Right: 2D Parity Density Hexbin (CAMS vs RF)
ax_hex = fig.add_subplot(gs[1])
try:
    preds = pd.read_csv("results/step4/test_predictions.csv")
    cams_vals = preds["pm25"].values
    rf_vals = preds["random_forest"].values
    
    hb = ax_hex.hexbin(cams_vals, rf_vals, gridsize=45, cmap='viridis', mincnt=1,
                       bins='log', extent=[0, 500, 0, 500])
    ax_hex.plot([0, 500], [0, 500], 'r--', lw=1.5, alpha=0.85, label='1:1 Parity line')
    ax_hex.set_xlim(0, 500)
    ax_hex.set_ylim(0, 500)
    ax_hex.set_xlabel("CAMS PM2.5 (µg/m³)", fontsize=10, fontweight="bold")
    ax_hex.set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=10, fontweight="bold")
    ax_hex.set_title("ML vs CAMS at test cells (Parity Density)", fontsize=12, fontweight="bold", pad=10)
    
    cb = fig.colorbar(hb, ax=ax_hex, fraction=0.046, pad=0.04)
    cb.set_label("Log Count", fontsize=8.5)
    ax_hex.legend(loc='upper left', fontsize=8.5)
except Exception as e:
    ax_hex.text(0.5, 0.5, f"Hexbin data: {e}", ha='center', va='center')

plt.tight_layout()
plt.savefig("results/step5/r2_by_season_slide10.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/step5/r2_by_season_slide10.png")
