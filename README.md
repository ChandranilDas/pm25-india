# Spatial Enhancement of Daily PM2.5 over India: Physics-Informed Machine Learning & Deep Residual Tabular Networks

[![Live Demo](https://img.shields.io/badge/Live%20Platform-pm25--india.vercel.app-0070F3?logo=vercel&style=flat-square)](https://pm25-india.vercel.app)
[![IEEE Paper](https://img.shields.io/badge/IEEE%20Format-Report.pdf-blue?style=flat-square)](Report.pdf)
[![Presentation Slides](https://img.shields.io/badge/Presentation-5%20Slides-orange?style=flat-square)](presentation_5slides.pptx)
[![Interactive Dashboard](https://img.shields.io/badge/Live%20Studio-Web%20Dashboard-emerald?style=flat-square)](https://pm25-india.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

> 🌐 **Live Web Application**: Access the full interactive platform, Leaflet GIS, real-time Open-Meteo weather inference, and multi-model benchmark predictions at **[https://pm25-india.vercel.app](https://pm25-india.vercel.app)**.

**EE708: Fundamentals of Data Science and Machine Intelligence Term Project (Group 1 • Project 3)**

An end-to-end atmospheric spatial enhancement system that estimates daily surface particulate matter ($\text{PM}_{2.5}$) across the Indian subcontinent (2020–2024, 902,538 daily records) from ECMWF Copernicus Atmosphere Monitoring Service (CAMS) reanalysis and ERA5 meteorological drivers.

The project evaluates a **multi-architecture benchmark** across **112 held-out spatial test cells** (zero spatial autocorrelation leakage) and introduces **AdvectSR-PM** (Physics-Informed Machine Learning with mass conservation) and **ResTabNet** (Parameter-Efficient Deep Tabular Residual Networks).

---

## 📊 Benchmark Results (112 Strictly Held-Out Spatial Test Cells)

Evaluating 204,624 daily test samples across 112 unmonitored cells (5 geographic blocks of $5\times 5$ cells):

| Architecture | Model Parameters / Footprint | Test $R^2$ | Overall RMSE ($\mu\text{g}/\text{m}^3$) | Tail RMSE ($>150\,\mu\text{g}/\text{m}^3$) | Severe Episode F1 | Moran's $I$ (Error Autocorr.) | Coarse Mass Consistency Error (CCE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 14 (<1 KB) | 0.4605 | 46.12 | 106.21 | 0.474 | 0.6052 | None |
| **IDW Spatial Interpolation** | Non-parametric | 0.6830 | 35.34 | 88.42 | 0.582 | 0.5120 | None |
| **Random Forest (Baseline)** | 100 trees (~1.5M nodes, 166.0 MB) | 0.7421 | 31.89 | 79.55 | 0.730 | 0.4374 | None |
| **HistGradientBoosting (HGB)** | 200 binned trees (340.8 KB) | 0.7541 | 31.14 | 77.85 | 0.746 | 0.3288 | None |
| **AdvectSR-PM (Physics PIML)** | Graph Advection PDE (865.9 KB) | 0.7768 | 29.67 | 74.00 | 0.768 | 0.2944 | **$\text{CCE} \equiv 0.000\,\mu\text{g}/\text{m}^3$** |
| **Advect-ResTabNet (Hybrid)** | PIML + Deep Learning (1.60 MB) | **0.8254** | **26.24** | **64.65** | **0.802** | **0.2922** | **$\text{CCE} \equiv 0.000\,\mu\text{g}/\text{m}^3$** |
| **ResTabNet (Proposed Neural)**| 58,881 weights (773.6 KB, Huber $\delta=1$) | **0.8274** | **26.09** | **63.31** | **0.803** | **0.2840** | $0.082\,\mu\text{g}/\text{m}^3$ |

---

## 🔬 Core Methodological Innovations

1. **AdvectSR-PM (Physics-Informed Atmospheric Advection)**:
   - Embeds the governing 2D atmospheric continuity PDE:
     $$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u} C) = \nabla \cdot (K \nabla C) + Q - \lambda C$$
   - Enforces preferential downwind information propagation along horizontal wind vectors $\mathbf{u} = [u_{10}, v_{10}]$ via dynamic graph adjacency $A_{ij,t}$.
   - Guarantees exact Coarse Consistency ($\text{CCE} \equiv 0.000\,\mu\text{g}/\text{m}^3$) by non-negative area-weighted normalization.
   - Outperforms all baselines in the rugged **Himalayan Basin ($R^2 = 0.860$)** and **Northeast Valleys ($R^2 = 0.798$)**, where IDW collapses ($R^2 = -0.740$).
2. **ResTabNet (Deep Tabular Residual Highway)**:
   - 58,881 weights with Huber loss ($\delta=1.0$), avoiding gradient explosion during intense stubble-burning episodes.
   - 214× smaller storage footprint than Random Forest (773.6 KB vs 166.0 MB) while improving out-of-sample $R^2$ by $+0.085$.
3. **53.1% Residual Spatial Autocorrelation Reduction**:
   - Moran's $I$ on test errors drops from $0.6052 \to 0.2840$, confirming error spatial independence.
4. **Tail-Risk Performance ($PM_{2.5} \ge 150\,\mu\text{g}/\text{m}^3$)**:
   - Tail RMSE drops from $106.21 \to 63.31\,\mu\text{g}/\text{m}^3$, with severe episode F1 reaching $0.803$ (77.0% recall).

---

## ⚡ Interactive Web Dashboard & Vercel Hosting

The web interface is available at `index.html` (or `web_dashboard.html`), featuring:
- **Interactive Multi-Layer Leaflet GIS Map**: Click or drag pin across India; syncs with 494 CAMS grid climatologies and 115+ Indian cities.
- **Live Open-Meteo Weather Integration**: Fetches real-time temperature, dewpoint, pressure, and wind vectors for any Indian coordinate.
- **5-Architecture Real-Time Comparison**: Live inference across ResTabNet, AdvectSR-PM, Advect-ResTabNet Hybrid, HistGradientBoosting, and Random Forest with 90% confidence intervals.
- **CPCB GRAP Stage I–IV Health Protocol**: Automated health advisories and WHO 24-hour exceedance factors.
- **Historical Crisis Replay Presets**: 1-click simulations for 2021 Diwali smog, 2023 stubble burning, 2020 lockdown baseline, 2022 monsoon washout, 2024 Thar dust storm, and national baseline.
- **Explainable AI (SHAP Proxy Waterfall)**: Real-time atmospheric feature attribution deltas.
- **Interactive Live Jupyter Notebook Viewer**: Direct inspection of all 19 `Code.ipynb` cells, outputs, and embedded figures.

---

## 🚀 Quickstart & Local Execution

### 1. Launch the Live Web Studio:
```bash
# Start backend daemon
python local_server.py
# Open in browser: http://localhost:8080
```

### 2. Run the Standardized Evaluation Pipeline:
```bash
# Evaluate all architectures on held-out test cells
python evaluate.py --model all --data data/pm25_daily.csv

# Evaluate individual models
python evaluate.py --model hybrid
python evaluate.py --model advectsr
python evaluate.py --model restabnet
python evaluate.py --model hgb
python evaluate.py --model rf
```

### 3. Deploy to Vercel:
The repository is pre-configured with `vercel.json` and standalone `index.html`. Deploy with one command:
```bash
vercel --prod
```

---

## 📁 Repository Structure

```
├── index.html                   # Vercel entrypoint (Prism/Leaflet full dashboard)
├── web_dashboard.html           # Standalone interactive explorer
├── vercel.json                  # Vercel deployment routes & configuration
├── local_server.py              # Python HTTP server backend & inference API
├── evaluate.py                  # Standalone multi-architecture evaluation pipeline
├── rf_worker.py                 # Multi-model IPC inference worker (RF, HGB, AdvectSR)
├── pm25_model.h5                # Trained ResTabNet neural network (58k params, 773 KB)
├── Code.ipynb                   # Executable Jupyter Notebook with pre-rendered slide plots
├── Report.pdf                   # Official IEEE Conference Paper (strictly 3 pages)
├── Report.docx                  # Synchronized Word format report
├── Report.tex                   # LaTeX paper source code
├── presentation_5slides.pptx    # Synchronized 5-slide executive presentation
├── results/
│   ├── step4/
│   │   ├── advectsr_model.joblib # AdvectSR-PM PIML model (866 KB)
│   │   ├── hgb_model.joblib      # HistGradientBoosting model (340 KB)
│   │   └── norm_params.npz       # Input scaling mean and variance vectors
│   ├── step5/
│   │   └── tail_and_moran_metrics.json # Formal tail risk & Moran's I diagnostics
│   └── step6/
│       └── india_025deg_pm25_grid.csv # High-res 0.25° fine grid data export (124 KB)
└── .gitignore                   # Excludes heavy datasets (>100MB) for GitHub compliance
```

---

## 📜 Academic Integrity & Citation

This project was developed for **EE708: Fundamentals of Data Science and Machine Intelligence** at the Department of Electrical Engineering. All training and evaluation adhere to strict spatial cross-validation isolation to ensure unbiased out-of-sample generalization.
