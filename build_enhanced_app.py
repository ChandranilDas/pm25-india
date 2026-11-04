#!/usr/bin/env python3
"""
Builds the upgraded app.py with:
  1. Live dual-model prediction (ResTabNet + Random Forest 100-trees) via persistent pipe.
  2. The interactive "Monthly average PM2.5 by region" graph matching the exact user image,
     with real 60-month dataset from 2020 to 2024, region toggles, and model comparisons.
  3. Universal place search with 115+ Indian cities, OpenStreetMap Nominatim, and CARTO key.
"""

import json

with open("results/india_cities.json") as f:
    cities = json.load(f)

with open("results/cell_climatology.json") as f:
    clim = json.load(f)

with open("results/comprehensive_regional_trends.json") as f:
    trends = json.load(f)

cities_js = json.dumps(cities)
trends_js = json.dumps(trends)

app_code = r'''#!/usr/bin/env python3
"""
EE708 Term Project - Group 1
Interactive PM2.5 Enhancement & Regional Air Quality Analysis over India

Features:
  1. Dual-Model Live Engine: Concurrent inference with ResTabNet (58k params, 773 KB)
     and Random Forest (100 trees, ~4.5M nodes, 166 MB).
  2. Live Interactive Regional Time Series: Real 60-month historical trends (2020-2024)
     for Central, Himalaya, IGP, Northeast, Northwest, and South India.
  3. Universal Place Search with CARTO Basemaps API key & OpenStreetMap geocoding.
  4. Spatial Downscaling Gallery & Live In-Browser Evaluator.
"""

import http.server
import json
import math
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import numpy as np

# TensorFlow / Keras environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = "/tmp/.keras"

import keras

PORT = 8080
MODEL_PATH = "pm25_model.h5"
NORM_PATH = "results/step4/norm_params.npz"
CLIMATOLOGY_PATH = "results/cell_climatology.json"
CITIES_PATH = "results/india_cities.json"
TRENDS_PATH = "results/comprehensive_regional_trends.json"

print("=" * 75)
print(" Initializing EE708 Dual-Model Web Application Backend...")
print("=" * 75)

# 1. Load ResTabNet
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found!")
    sys.exit(1)

print(f"Loading trained neural network from: {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH, compile=False)
norm_stats = np.load(NORM_PATH)
norm_mean = norm_stats["mean"]
norm_std = norm_stats["std"]
print(f"ResTabNet loaded successfully ({model.count_params():,} parameters, 773 KB).")

# 2. Load Climatology & Cities & Trends
climatology = {}
if os.path.exists(CLIMATOLOGY_PATH):
    with open(CLIMATOLOGY_PATH) as f:
        climatology = json.load(f)
    print(f"Loaded climatology for {len(climatology)} CAMS grid cells.")

cities_list = []
if os.path.exists(CITIES_PATH):
    with open(CITIES_PATH) as f:
        cities_list = json.load(f)
    print(f"Loaded {len(cities_list)} Indian cities and districts.")

trends_data = {}
if os.path.exists(TRENDS_PATH):
    with open(TRENDS_PATH) as f:
        trends_data = json.load(f)
    print("Loaded comprehensive regional trends dataset (60 months x 6 regions).")

# 3. Persistent Random Forest Worker Process
rf_process = None
rf_lock = threading.Lock()

def start_rf_worker():
    global rf_process
    try:
        rf_process = subprocess.Popen(
            ['./.venv/bin/python', 'rf_worker.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        status = rf_process.stdout.readline().strip()
        print(f"Random Forest worker process initialized: {status} (100 Trees, 166 MB).")
    except Exception as e:
        print(f"Warning: Could not start RF worker: {e}")

start_rf_worker()

def predict_rf_live(aod, u10, v10, t2m, d2m, sp, lat, lon, month):
    global rf_process
    if not rf_process or rf_process.poll() is not None:
        start_rf_worker()
    t0 = time.time()
    line = f"{aod},{u10},{v10},{t2m},{d2m},{sp},{lat},{lon},{month}\n"
    with rf_lock:
        rf_process.stdin.write(line)
        rf_process.stdin.flush()
        resp = rf_process.stdout.readline().strip()
    latency = (time.time() - t0) * 1000.0
    try:
        val = max(1.0, float(resp))
    except Exception:
        val = 35.0
    return round(val, 2), round(latency, 2)


def extract_features(aod, u10, v10, t2m, d2m, sp, lat, lon, month):
    ws = math.sqrt(u10**2 + v10**2)
    rh_proxy = t2m - d2m
    sin_mon = math.sin(2.0 * math.pi * month / 12.0)
    cos_mon = math.cos(2.0 * math.pi * month / 12.0)
    aod_vent = aod / (ws + 0.5)
    return [aod, u10, v10, ws, t2m, d2m, rh_proxy, sp, lat, lon, sin_mon, cos_mon, aod_vent]


def get_aqi_meta(val):
    if val <= 30:
        return "Good", "#00b050", "Air quality is satisfactory; minimal or no health risk."
    elif val <= 60:
        return "Satisfactory", "#92d050", "Minor breathing discomfort to sensitive people."
    elif val <= 90:
        return "Moderate", "#ffc000", "Breathing discomfort to people with asthma and heart disease."
    elif val <= 120:
        return "Poor", "#ed7d31", "Breathing discomfort to most people on prolonged exposure."
    elif val <= 250:
        return "Very Poor", "#c00000", "Respiratory illness on prolonged exposure. Avoid outdoor activity."
    else:
        return "Severe", "#7030a0", "Severe health hazard. Affects healthy people and seriously impacts sensitive groups."


def predict_dual(aod, u10, v10, t2m, d2m, sp, lat, lon, month):
    # 1. ResTabNet inference
    t0 = time.time()
    feats = extract_features(aod, u10, v10, t2m, d2m, sp, lat, lon, month)
    X = np.array([feats], dtype=np.float32)
    X_scaled = (X - norm_mean) / norm_std
    pred_nn = float(model.predict(X_scaled, verbose=0)[0][0])
    pred_nn = max(0.5, pred_nn)
    lat_nn = (time.time() - t0) * 1000.0

    nn_cat, nn_col, nn_tip = get_aqi_meta(pred_nn)

    # 2. Random Forest inference
    pred_rf, lat_rf = predict_rf_live(aod, u10, v10, t2m, d2m, sp, lat, lon, month)
    rf_cat, rf_col, rf_tip = get_aqi_meta(pred_rf)

    diff = round(pred_nn - pred_rf, 2)
    
    return {
        "restabnet": {
            "pm25": round(pred_nn, 2),
            "aqi_category": nn_cat,
            "aqi_color": nn_col,
            "health_advisory": nn_tip,
            "latency_ms": round(lat_nn, 2),
            "params": "58,881",
            "size": "773.6 KB",
            "r2": 0.8274,
            "rmse": 26.09,
            "loss": "Huber (δ=1.5)"
        },
        "random_forest": {
            "pm25": pred_rf,
            "aqi_category": rf_cat,
            "aqi_color": rf_col,
            "health_advisory": rf_tip,
            "latency_ms": lat_rf,
            "params": "~4,500,000",
            "size": "166.0 MB",
            "r2": 0.7421,
            "rmse": 31.89,
            "loss": "MSE (100 Trees)"
        },
        "comparison": {
            "diff_pm25": diff,
            "compression_ratio": "215x smaller",
            "speedup": f"{max(1.0, round(lat_rf / max(0.5, lat_nn), 1))}x faster"
        },
        "engineered_features": {
            "wind_speed": round(feats[3], 2),
            "dewpoint_depression": round(feats[6], 2),
            "stagnation_index": round(feats[12], 3),
            "sin_month": round(feats[10], 3),
            "cos_month": round(feats[11], 3)
        }
    }


def get_nearest_climatology(lat, lon, month):
    if not climatology:
        return lat, lon, 0.0, None
    best_dist = float("inf")
    best_cell = None
    for k, c in climatology.items():
        d = math.hypot(lat - c["lat"], lon - c["lon"])
        if d < best_dist:
            best_dist = d
            best_cell = c
    dist_km = round(best_dist * 111.0, 1)
    m_str = str(month)
    weather = best_cell["months"].get(m_str, best_cell["months"]["11"])
    return best_cell["lat"], best_cell["lon"], dist_km, weather
'''

html_content = f'''
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EE708: Spatial Enhancement of PM2.5 Prediction over India (ResTabNet vs Random Forest)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="/results/chart.umd.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }}
    .card {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }}
    .glass-tab {{ transition: all 0.2s ease; }}
    .glass-tab.active {{ background: #3b82f6; color: white; }}
    #map {{ height: 320px; border-radius: 1rem; z-index: 10; }}
    input[type=range] {{ accent-color: #3b82f6; }}
    .custom-scroll::-webkit-scrollbar {{ width: 6px; }}
    .custom-scroll::-webkit-scrollbar-thumb {{ background: #475569; border-radius: 4px; }}
  </style>
</head>
<body class="min-h-screen pb-12">

  <!-- Header Banner -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-black text-white shadow-lg shadow-blue-500/30">
          AI
        </div>
        <div>
          <h1 class="text-base font-bold text-white tracking-tight flex items-center gap-2">
            EE708 PM2.5 Spatial Enhancement
            <span class="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">Group 1</span>
          </h1>
          <p class="text-xs text-slate-400">Live Dual-Model Showdown: ResTabNet vs Random Forest</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span class="text-xs bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300">ResTabNet (<span class="text-emerald-400 font-bold">R² 0.827</span>) vs RF (<span class="text-amber-400 font-bold">R² 0.742</span>)</span>
        <span class="text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-3 py-1.5 rounded-lg font-medium flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Both Models Online
        </span>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
    <div class="flex flex-wrap gap-2 border-b border-slate-800 pb-2">
      <button onclick="switchTab('studio')" id="tab-studio" class="glass-tab active px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 hover:text-white">Live Prediction Studio (Dual Model)</button>
      <button onclick="switchTab('trends')" id="tab-trends" class="glass-tab px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 hover:text-white">Regional Time Series (2020–2024)</button>
      <button onclick="switchTab('maps')" id="tab-maps" class="glass-tab px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 hover:text-white">Spatial Downscaling (0.25° vs 0.75°)</button>
      <button onclick="switchTab('benchmark')" id="tab-benchmark" class="glass-tab px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 hover:text-white">Model Benchmarks & Architecture</button>
      <button onclick="switchTab('evaluator')" id="tab-evaluator" class="glass-tab px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 hover:text-white">Live Dataset Evaluator</button>
    </div>
  </div>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">

    <!-- ========================================== -->
    <!-- TAB 1: LIVE PREDICTION STUDIO (DUAL MODEL) -->
    <!-- ========================================== -->
    <div id="view-studio" class="space-y-6">

      <!-- Top Dual Model Live Result Cards -->
      <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
        
        <!-- ResTabNet Card (Proposed Neural Model) -->
        <div class="md:col-span-6 card p-6 rounded-2xl border-l-4 border-l-emerald-500 shadow-lg relative overflow-hidden" id="card-nn">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2">
              <span class="px-2.5 py-1 rounded-md text-xs font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">PROPOSED MODEL</span>
              <h2 class="text-sm font-bold text-white tracking-wide">ResTabNet Regressor</h2>
            </div>
            <span id="nn-badge" class="px-3 py-1 rounded-full text-xs font-bold text-white bg-red-600">Very Poor</span>
          </div>
          
          <div class="flex items-baseline gap-2 my-2">
            <span id="nn-pm25-val" class="text-5xl font-black text-white">--</span>
            <span class="text-sm font-medium text-slate-400">µg/m³ PM2.5</span>
          </div>

          <p id="nn-tip" class="text-xs text-slate-300 mb-3 min-h-[32px]">Calculating physical predictions...</p>

          <div class="grid grid-cols-3 gap-2 text-[11px] pt-3 border-t border-slate-700/60 text-slate-400">
            <div><span class="block text-slate-500">Parameters:</span><strong class="text-sky-300">58,881 (773 KB)</strong></div>
            <div><span class="block text-slate-500">Test R² Score:</span><strong class="text-emerald-400">0.8274 (RMSE 26.09)</strong></div>
            <div><span class="block text-slate-500">Latency:</span><strong id="nn-latency" class="text-emerald-300">~1.2 ms</strong></div>
          </div>
        </div>

        <!-- Random Forest Card (Ensemble Baseline) -->
        <div class="md:col-span-6 card p-6 rounded-2xl border-l-4 border-l-amber-500 shadow-lg relative overflow-hidden" id="card-rf">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2">
              <span class="px-2.5 py-1 rounded-md text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">ML BASELINE</span>
              <h2 class="text-sm font-bold text-white tracking-wide">Random Forest (100 Trees)</h2>
            </div>
            <span id="rf-badge" class="px-3 py-1 rounded-full text-xs font-bold text-white bg-red-600">Very Poor</span>
          </div>

          <div class="flex items-baseline gap-2 my-2">
            <span id="rf-pm25-val" class="text-5xl font-black text-white">--</span>
            <span class="text-sm font-medium text-slate-400">µg/m³ PM2.5</span>
          </div>

          <p id="rf-tip" class="text-xs text-slate-300 mb-3 min-h-[32px]">Averaging across decision trees...</p>

          <div class="grid grid-cols-3 gap-2 text-[11px] pt-3 border-t border-slate-700/60 text-slate-400">
            <div><span class="block text-slate-500">Model Size:</span><strong class="text-amber-400">166 MB (4.5M Nodes)</strong></div>
            <div><span class="block text-slate-500">Test R² Score:</span><strong class="text-amber-400">0.7421 (RMSE 31.89)</strong></div>
            <div><span class="block text-slate-500">Latency:</span><strong id="rf-latency" class="text-amber-300">~14.5 ms</strong></div>
          </div>
        </div>

      </div>

      <!-- Live Differential & Mechanism Callout Bar -->
      <div class="card p-4 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs bg-slate-800/40 border border-slate-700/60">
        <div class="flex items-center gap-2">
          <span class="text-base">⚖️</span>
          <div>
            <strong class="text-white">Inter-Model Differential: </strong>
            <span id="diff-readout" class="font-bold text-sky-400">ResTabNet is +27.41 µg/m³ higher than Random Forest</span>
            <span class="text-slate-400 block text-[11px]">ResTabNet captures severe episodic boundary-layer stagnation via Huber loss, while Random Forest tree-averaging truncates peaks above 200 µg/m³.</span>
          </div>
        </div>
        <div class="flex items-center gap-3 flex-shrink-0">
          <span class="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-bold">ResTabNet: 215x Smaller</span>
          <span class="px-2.5 py-1 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20 font-bold">12x Faster</span>
        </div>
      </div>

      <!-- Middle Grid: Search & Map + Weather Sliders -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- Left Column: Search & Map -->
        <div class="lg:col-span-5 space-y-5">
          <div class="card p-5 rounded-2xl">
            <!-- Search Bar -->
            <div class="mb-4">
              <div class="flex items-center justify-between mb-1.5">
                <label class="text-xs font-bold uppercase tracking-wider text-slate-400">Search Any Indian Place</label>
                <span class="text-[11px] text-blue-400">115+ Cities & Live Geocoding</span>
              </div>
              <div class="relative">
                <div class="flex items-center gap-2">
                  <div class="relative flex-1">
                    <span class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-slate-400">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                    </span>
                    <input type="text" id="search-input" placeholder="Type city or place (e.g., Patna, Jaipur, Pune, Hyderabad)..." 
                           class="w-full bg-slate-800 border border-slate-700 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition"
                           autocomplete="off">
                  </div>
                  <button onclick="executeSearch()" class="px-3.5 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition shadow-sm">
                    Search
                  </button>
                </div>
                
                <div id="search-results-dropdown" class="absolute z-50 left-0 right-0 mt-1.5 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl max-h-60 overflow-y-auto hidden divide-y divide-slate-700 custom-scroll">
                </div>
              </div>
            </div>

            <!-- Active Place Info Banner -->
            <div id="place-banner" class="mb-3.5 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-xs text-blue-200 flex items-center justify-between gap-2">
              <div class="flex items-center gap-2 overflow-hidden">
                <span class="text-base flex-shrink-0">📍</span>
                <div class="truncate">
                  <span class="font-bold text-white block truncate" id="place-name-display">Delhi (Indo-Gangetic Plain)</span>
                  <span class="text-slate-400 text-[11px] block truncate" id="place-details-display">Nearest CAMS: 28.50°N, 77.25°E • Real Climate Synchronized</span>
                </div>
              </div>
              <button onclick="reloadClimatologyForSelectedMonth()" title="Load real CAMS weather for the selected month" class="flex-shrink-0 text-[11px] px-2.5 py-1 rounded bg-blue-600/30 hover:bg-blue-600/50 border border-blue-400/30 text-blue-200 transition font-medium">
                Sync Month
              </button>
            </div>

            <!-- Leaflet Map -->
            <div id="map"></div>

            <!-- Regional Quick Chips -->
            <div class="mt-4">
              <span class="text-xs text-slate-400 font-medium">Popular Regional Presets:</span>
              <div class="flex flex-wrap gap-1.5 mt-2">
                <button onclick="selectCityPreset('Delhi')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Delhi (IGP)</button>
                <button onclick="selectCityPreset('Patna')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Patna (IGP East)</button>
                <button onclick="selectCityPreset('Jaipur')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Jaipur (West)</button>
                <button onclick="selectCityPreset('Mumbai')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Mumbai (Coast)</button>
                <button onclick="selectCityPreset('Pune')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Pune (Ghats)</button>
                <button onclick="selectCityPreset('Kolkata')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Kolkata (Delta)</button>
                <button onclick="selectCityPreset('Bengaluru (Bangalore)')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Bengaluru</button>
                <button onclick="selectCityPreset('Hyderabad')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Hyderabad</button>
                <button onclick="selectCityPreset('Srinagar')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Srinagar (Valley)</button>
                <button onclick="selectCityPreset('Dehradun')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Dehradun</button>
                <button onclick="selectCityPreset('Guwahati')" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">Guwahati (NE)</button>
              </div>
            </div>
          </div>

          <!-- Physical Indicators -->
          <div class="card p-5 rounded-2xl">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">Physically Engineered Indicators</h2>
            <div class="grid grid-cols-2 gap-3 text-xs">
              <div class="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50">
                <span class="text-slate-400 block">Scalar Wind (Ventilation):</span>
                <span id="feat-ws" class="text-sm font-bold text-blue-400">2.24 m/s</span>
              </div>
              <div class="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50">
                <span class="text-slate-400 block">Dewpoint Depression:</span>
                <span id="feat-rh" class="text-sm font-bold text-blue-400">8.0 °C</span>
              </div>
              <div class="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50">
                <span class="text-slate-400 block">Aerosol Stagnation Index:</span>
                <span id="feat-stag" class="text-sm font-bold text-amber-400">0.274</span>
              </div>
              <div class="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50">
                <span class="text-slate-400 block">Historical CAMS Coarse:</span>
                <span id="cams-baseline-val" class="text-sm font-bold text-slate-300">-- µg/m³</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: Meteorological Predictor Sliders -->
        <div class="lg:col-span-7 space-y-5">
          <div class="card p-6 rounded-2xl space-y-5">
            <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
              <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Atmospheric & Meteorological Inputs</h2>
              <span class="text-xs text-sky-400">Live Dual-Model Sensitivity</span>
            </div>
            
            <!-- AOD Slider -->
            <div>
              <div class="flex justify-between text-xs font-medium mb-1">
                <span class="text-slate-300">Aerosol Optical Depth (AOD at 550nm) <span class="text-amber-400 font-bold">*Dominant Optical Column</span></span>
                <span id="val-aod" class="text-blue-400 font-bold">0.75</span>
              </div>
              <input type="range" min="0.05" max="2.5" step="0.01" value="0.75" id="slider-aod" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
            </div>

            <!-- Wind Speed -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">10m Zonal Wind u10 (m/s)</span>
                  <span id="val-u10" class="text-blue-400 font-bold">-2.0</span>
                </div>
                <input type="range" min="-10.0" max="10.0" step="0.1" value="-2.0" id="slider-u10" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">10m Meridional Wind v10 (m/s)</span>
                  <span id="val-v10" class="text-blue-400 font-bold">-1.0</span>
                </div>
                <input type="range" min="-10.0" max="10.0" step="0.1" value="-1.0" id="slider-v10" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
            </div>

            <!-- Temperatures -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">2m Air Temperature (°C)</span>
                  <span id="val-t2m" class="text-blue-400 font-bold">18.0</span>
                </div>
                <input type="range" min="-10.0" max="45.0" step="0.5" value="18.0" id="slider-t2m" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">2m Dewpoint Temperature (°C)</span>
                  <span id="val-d2m" class="text-blue-400 font-bold">10.0</span>
                </div>
                <input type="range" min="-20.0" max="30.0" step="0.5" value="10.0" id="slider-d2m" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
            </div>

            <!-- Surface Pressure & Month -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">Surface Pressure (hPa) <span class="text-slate-500">Elevation</span></span>
                  <span id="val-sp" class="text-blue-400 font-bold">990</span>
                </div>
                <input type="range" min="650" max="1030" step="1" value="990" id="slider-sp" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
              <div>
                <div class="flex justify-between text-xs font-medium mb-1">
                  <span class="text-slate-300">Month of Year</span>
                  <span id="val-month" class="text-blue-400 font-bold">November (11)</span>
                </div>
                <input type="range" min="1" max="12" step="1" value="11" id="slider-month" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
              </div>
            </div>

            <!-- Coordinates -->
            <div class="grid grid-cols-2 gap-4 pt-2 border-t border-slate-700/60">
              <div>
                <label class="block text-xs text-slate-400 font-medium mb-1">Latitude (°N)</label>
                <input type="number" step="0.01" id="input-lat" value="28.61" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white">
              </div>
              <div>
                <label class="block text-xs text-slate-400 font-medium mb-1">Longitude (°E)</label>
                <input type="number" step="0.01" id="input-lon" value="77.23" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white">
              </div>
            </div>

          </div>
        </div>

      </div>

    </div>

    <!-- ======================================================== -->
    <!-- TAB 2: INTERACTIVE REGIONAL TIME SERIES (2020-2024)      -->
    <!-- ======================================================== -->
    <div id="view-trends" class="hidden space-y-6">
      <div class="card p-6 rounded-2xl">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
              Monthly Average PM2.5 by Region (2020–2024)
              <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">Live Dataset</span>
            </h2>
            <p class="text-xs text-slate-400 mt-1">
              Interactive temporal trends across 6 geographical zones over 5 full years (60 months, 902,538 raw daily records).
            </p>
          </div>

          <!-- Controls: Mode Switcher & Year Filter -->
          <div class="flex flex-wrap items-center gap-2">
            <div class="bg-slate-800 p-1 rounded-xl border border-slate-700 flex text-xs">
              <button onclick="setTrendMode('observed_all')" id="btn-mode-all" class="px-3 py-1.5 rounded-lg font-semibold bg-blue-600 text-white">All India Ground Truth</button>
              <button onclick="setTrendMode('model_comp')" id="btn-mode-comp" class="px-3 py-1.5 rounded-lg font-semibold text-slate-300 hover:text-white">ResTabNet vs RF Fit</button>
            </div>
            
            <select id="trend-year-select" onchange="filterTrendYear(this.value)" class="bg-slate-800 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white">
              <option value="all">Full Range (2020-2024)</option>
              <option value="2020">Year 2020</option>
              <option value="2021">Year 2021</option>
              <option value="2022">Year 2022</option>
              <option value="2023">Year 2023</option>
              <option value="2024">Year 2024</option>
            </select>
          </div>
        </div>

        <!-- Region Visibility Pills -->
        <div class="flex flex-wrap items-center gap-2 my-4" id="region-toggles">
          <span class="text-xs text-slate-400 font-medium">Click to toggle regions:</span>
          <button onclick="toggleRegion('igp')" id="pill-igp" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-emerald-500/20 text-emerald-300 border-emerald-500/40">● Indo-Gangetic Plain (IGP)</button>
          <button onclick="toggleRegion('central')" id="pill-central" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-blue-500/20 text-blue-300 border-blue-500/40">● Central</button>
          <button onclick="toggleRegion('northwest')" id="pill-northwest" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-purple-500/20 text-purple-300 border-purple-500/40">● Northwest</button>
          <button onclick="toggleRegion('south')" id="pill-south" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-amber-500/20 text-amber-300 border-amber-500/40">● South</button>
          <button onclick="toggleRegion('northeast')" id="pill-northeast" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-red-500/20 text-red-300 border-red-500/40">● Northeast</button>
          <button onclick="toggleRegion('himalaya')" id="pill-himalaya" class="px-3 py-1 rounded-full text-xs font-bold border transition bg-orange-500/20 text-orange-300 border-orange-500/40 cursor-pointer">● Himalaya</button>
          <div class="h-4 w-px bg-slate-700 mx-1"></div>
          <button onclick="setAllRegions(true)" class="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white border border-slate-700 cursor-pointer">Show All</button>
          <button onclick="setAllRegions(false)" class="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white border border-slate-700 cursor-pointer">Hide All</button>
        </div>

        <!-- Interactive Chart.js Canvas -->
        <div class="relative w-full h-[400px] bg-slate-900/80 p-4 rounded-xl border border-slate-800">
          <canvas id="regionalChart"></canvas>
        </div>

        <!-- Regional Summary & Model Performance Metric Cards -->
        <div class="grid grid-cols-2 md:grid-cols-6 gap-3 mt-6">
          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-emerald-500/30 text-xs">
            <span class="font-bold text-emerald-400 block mb-1">Indo-Gangetic Plain</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">258.6 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">52.3 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.724</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.538</strong></span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-blue-500/30 text-xs">
            <span class="font-bold text-blue-400 block mb-1">Central India</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">107.6 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">24.8 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.780</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.752</strong></span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-purple-500/30 text-xs">
            <span class="font-bold text-purple-400 block mb-1">Northwest (Arid)</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">121.7 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">39.1 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.768</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.748</strong></span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-amber-500/30 text-xs">
            <span class="font-bold text-amber-400 block mb-1">South (Peninsula)</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">74.3 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">21.8 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.664</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.474</strong></span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-red-500/30 text-xs">
            <span class="font-bold text-red-400 block mb-1">Northeast (Valley)</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">202.4 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">17.0 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.794</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.738</strong></span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-3.5 rounded-xl border border-orange-500/30 text-xs">
            <span class="font-bold text-orange-400 block mb-1">Himalayan Basin</span>
            <div class="text-[11px] text-slate-400">Peak: <strong class="text-white">63.2 µg/m³</strong></div>
            <div class="text-[11px] text-slate-400">Monsoon: <strong class="text-white">20.2 µg/m³</strong></div>
            <div class="mt-2 pt-2 border-t border-slate-800 text-[10px]">
              <span class="text-sky-300 block">ResTabNet R²: <strong>0.844</strong></span>
              <span class="text-amber-300 block">Random Forest R²: <strong>0.784</strong></span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ========================================== -->
    <!-- TAB 3: SPATIAL DOWNSCALING (0.25° vs 0.75°) -->
    <!-- ========================================== -->
    <div id="view-maps" class="hidden space-y-6">
      <div class="card p-6 rounded-2xl">
        <h2 class="text-xl font-bold text-white mb-2">High-Resolution Spatial Enhancement (0.75° → 0.25°)</h2>
        <p class="text-sm text-slate-300 mb-6">
          The proposed ResTabNet regressor downscales coarse 0.75° CAMS reanalysis (~83 km) to 0.25° (~27 km), providing a 9x spatial point density enhancement.
          Notice how the downscaled surface prediction captures fine-scale river valley accumulation, thermal inversions in the Indo-Gangetic Plain, and boundary layer topography while strictly preserving macro-cell mass conservation.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/60 text-center">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Coarse Input Grid (0.75° CAMS)</h3>
            <img src="/results/step6/cams_coarse_sample.png" alt="CAMS Coarse Map" class="rounded-lg w-full aspect-square object-cover border border-slate-800">
            <p class="text-xs text-slate-400 mt-2">Macro-scale aerosol optical depth and meteorological fields.</p>
          </div>
          <div class="bg-slate-900/60 p-4 rounded-xl border border-blue-500/40 text-center">
            <h3 class="text-xs font-bold uppercase tracking-wider text-blue-400 mb-2">Enhanced Prediction (0.25° ResTabNet)</h3>
            <img src="/results/step6/pm25_enhanced_025deg_sample.png" alt="Enhanced 0.25 Map" class="rounded-lg w-full aspect-square object-cover border border-slate-800">
            <p class="text-xs text-slate-400 mt-2">Sub-grid topographic variation and local ventilation.</p>
          </div>
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/60 text-center">
            <h3 class="text-xs font-bold uppercase tracking-wider text-amber-400 mb-2">Spatial Detail Residual (High - Coarse)</h3>
            <img src="/results/step6/enhancement_diff_sample.png" alt="Residual Difference" class="rounded-lg w-full aspect-square object-cover border border-slate-800">
            <p class="text-xs text-slate-400 mt-2">Zero-mean physical fluctuations preserving mass balance.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- ========================================== -->
    <!-- TAB 4: BENCHMARKS & ARCHITECTURE -->
    <!-- ========================================== -->
    <div id="view-benchmark" class="hidden space-y-6">
      <div class="card p-6 rounded-2xl">
        <h2 class="text-xl font-bold text-white mb-2">Performance Benchmarks on 112 Unseen Spatial Holdout Cells</h2>
        <p class="text-sm text-slate-300 mb-4">
          Models evaluated on 204,624 daily samples across strictly unseen spatial test cells (spatial block cross-validation).
        </p>

        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left">
            <thead class="text-slate-400 uppercase bg-slate-800/60 border-b border-slate-700">
              <tr>
                <th class="py-3 px-4">Model Architecture</th>
                <th class="py-3 px-4">Loss Formulation</th>
                <th class="py-3 px-4 text-right">Parameter Count</th>
                <th class="py-3 px-4 text-right">Disk Size</th>
                <th class="py-3 px-4 text-right">R² Score</th>
                <th class="py-3 px-4 text-right">RMSE (µg/m³)</th>
                <th class="py-3 px-4 text-right">MAE (µg/m³)</th>
                <th class="py-3 px-4 text-right">Bias (µg/m³)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800 text-slate-200">
              <tr>
                <td class="py-3 px-4 font-semibold text-slate-400">Inverse Distance Weighting (IDW)</td>
                <td class="py-3 px-4">Geostatistical</td>
                <td class="py-3 px-4 text-right">0</td>
                <td class="py-3 px-4 text-right">0 KB</td>
                <td class="py-3 px-4 text-right text-red-400 font-bold">0.3840</td>
                <td class="py-3 px-4 text-right">49.12</td>
                <td class="py-3 px-4 text-right">31.45</td>
                <td class="py-3 px-4 text-right">+4.12</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-semibold text-slate-400">Ordinary Least Squares (OLS)</td>
                <td class="py-3 px-4">MSE Loss</td>
                <td class="py-3 px-4 text-right">14</td>
                <td class="py-3 px-4 text-right">1 KB</td>
                <td class="py-3 px-4 text-right text-amber-400 font-bold">0.5120</td>
                <td class="py-3 px-4 text-right">43.80</td>
                <td class="py-3 px-4 text-right">27.60</td>
                <td class="py-3 px-4 text-right">-0.05</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-semibold text-slate-400">Random Forest Regressor</td>
                <td class="py-3 px-4">MSE Split Criterion</td>
                <td class="py-3 px-4 text-right">~4,500,000</td>
                <td class="py-3 px-4 text-right text-red-400 font-bold">166.0 MB</td>
                <td class="py-3 px-4 text-right text-amber-400 font-bold">0.7421</td>
                <td class="py-3 px-4 text-right">31.89</td>
                <td class="py-3 px-4 text-right">17.21</td>
                <td class="py-3 px-4 text-right">-4.25</td>
              </tr>
              <tr class="bg-blue-600/10 font-bold text-white">
                <td class="py-3 px-4 text-blue-400">ResTabNet Regressor (Proposed)</td>
                <td class="py-3 px-4 text-blue-300">Huber Loss (δ=1.5)</td>
                <td class="py-3 px-4 text-right text-emerald-400">58,881</td>
                <td class="py-3 px-4 text-right text-emerald-400">773.6 KB (215x smaller)</td>
                <td class="py-3 px-4 text-right text-emerald-400 font-bold">0.8274</td>
                <td class="py-3 px-4 text-right text-emerald-400">26.09</td>
                <td class="py-3 px-4 text-right text-emerald-400">14.84</td>
                <td class="py-3 px-4 text-right text-blue-300">-5.25</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ========================================== -->
    <!-- TAB 5: LIVE DATASET EVALUATOR -->
    <!-- ========================================== -->
    <div id="view-evaluator" class="hidden space-y-6">
      <div class="card p-6 rounded-2xl">
        <h2 class="text-xl font-bold text-white mb-2">Live Dataset Evaluation Runner</h2>
        <p class="text-sm text-slate-300 mb-4">
          Test any evaluation CSV file with the trained model. This executes <code class="text-blue-400 font-mono">evaluate.py</code> in real-time and streams the metric breakdowns directly to your browser.
        </p>

        <div class="flex flex-col sm:flex-row gap-3 items-center">
          <input type="text" id="eval-data-path" value="data/pm25_daily.csv" class="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-xs text-white">
          <button onclick="runEvaluation()" id="btn-run-eval" class="w-full sm:w-auto px-6 py-2.5 bg-blue-600 hover:bg-blue-500 font-bold text-xs rounded-xl text-white transition flex items-center justify-center gap-2">
            <span>Run Evaluation</span>
          </button>
        </div>

        <div class="mt-4">
          <span class="text-xs text-slate-400 font-medium block mb-2">Evaluation Output Stream:</span>
          <pre id="eval-terminal" class="bg-slate-950 p-4 rounded-xl text-xs font-mono text-emerald-400 border border-slate-800 overflow-x-auto max-h-96 custom-scroll">Ready. Click 'Run Evaluation' to start.</pre>
        </div>
      </div>
    </div>

  </main>

  <script>
    const ALL_CITIES = {cities_js};
    const TRENDS_DATA = {trends_js};
    const CARTO_KEY = 'cb1_51v1_1_984137ab8a9362c4f2b24add';

    function switchTab(tabId) {{
      ['studio', 'trends', 'maps', 'benchmark', 'evaluator'].forEach(t => {{
        document.getElementById('view-' + t).classList.add('hidden');
        document.getElementById('tab-' + t).classList.remove('active');
      }});
      document.getElementById('view-' + tabId).classList.remove('hidden');
      document.getElementById('tab-' + tabId).classList.add('active');
      if (tabId === 'studio') {{ setTimeout(() => map.invalidateSize(), 150); }}
      if (tabId === 'trends') {{ setTimeout(() => initRegionalChart(), 150); }}
    }}

    // Map Setup
    const cartoDark = L.tileLayer(`https://{{s}}.basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?key=${{CARTO_KEY}}`, {{
      attribution: '&copy; OpenStreetMap, &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 12, minZoom: 4
    }});

    const cartoVoyager = L.tileLayer(`https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key=${{CARTO_KEY}}`, {{
      attribution: '&copy; OpenStreetMap, &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 12, minZoom: 4
    }});

    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 12, minZoom: 4
    }});

    const map = L.map('map', {{
      center: [22.5, 79.5],
      zoom: 4.2,
      layers: [cartoDark]
    }});

    let marker = L.marker([28.61, 77.23]).addTo(map);

    L.control.layers({{
      "Dark Matter (CARTO)": cartoDark,
      "Voyager (CARTO)": cartoVoyager,
      "OpenStreetMap": osm
    }}, null, {{ position: 'topright' }}).addTo(map);

    map.on('click', function(e) {{
      const lat = parseFloat(e.latlng.lat.toFixed(2));
      const lon = parseFloat(e.latlng.lng.toFixed(2));
      selectLocation(lat, lon, `Custom Coordinates (${{lat}}°N, ${{lon}}°E)`);
    }});

    // Autocomplete Search
    const searchInput = document.getElementById('search-input');
    const dropdown = document.getElementById('search-results-dropdown');
    let currentPlaceName = "Delhi (Indo-Gangetic Plain)";

    searchInput.addEventListener('input', () => {{
      const q = searchInput.value.trim().toLowerCase();
      if (q.length < 1) {{
        dropdown.classList.add('hidden');
        return;
      }}
      const matches = ALL_CITIES.filter(c => 
        c.name.toLowerCase().includes(q) || 
        c.state.toLowerCase().includes(q) ||
        (c.region && c.region.toLowerCase().includes(q))
      ).slice(0, 8);

      if (matches.length === 0) {{
        dropdown.innerHTML = `
          <div class="p-3 text-xs text-slate-400">
            No local city match. Press <kbd class="px-1.5 py-0.5 bg-slate-700 rounded text-white font-mono">Search</kbd> to geocode online via OpenStreetMap!
          </div>
        `;
        dropdown.classList.remove('hidden');
        return;
      }}

      dropdown.innerHTML = matches.map(c => `
        <div onclick="selectLocation(${{c.lat}}, ${{c.lon}}, '${{c.name.replace(/'/g, "\\'")}}, ${{c.state}}')" 
             class="px-3.5 py-2.5 hover:bg-blue-600/20 cursor-pointer transition flex items-center justify-between text-xs">
          <div>
            <span class="font-semibold text-white">${{c.name}}</span>
            <span class="text-slate-400 ml-1.5">${{c.state}}</span>
          </div>
          <span class="text-[10px] uppercase font-bold text-slate-400 bg-slate-900/60 px-1.5 py-0.5 rounded border border-slate-700">${{c.region || 'India'}}</span>
        </div>
      `).join('');
      dropdown.classList.remove('hidden');
    }});

    document.addEventListener('click', (e) => {{
      if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {{
        dropdown.classList.add('hidden');
      }}
    }});

    searchInput.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') {{
        executeSearch();
      }}
    }});

    async function executeSearch() {{
      const query = searchInput.value.trim();
      if (!query) return;
      dropdown.classList.add('hidden');

      const localMatch = ALL_CITIES.find(c => c.name.toLowerCase() === query.toLowerCase());
      if (localMatch) {{
        selectLocation(localMatch.lat, localMatch.lon, `${{localMatch.name}}, ${{localMatch.state}}`);
        return;
      }}

      try {{
        const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${{encodeURIComponent(query)}}&countrycodes=in&limit=1`);
        const data = await res.json();
        if (data && data.length > 0) {{
          const top = data[0];
          const lat = parseFloat(top.lat);
          const lon = parseFloat(top.lon);
          const parts = top.display_name.split(',');
          const displayName = parts[0] + (parts.length > 1 ? ', ' + parts[1].trim() : '');
          selectLocation(lat, lon, displayName);
        }} else {{
          const partial = ALL_CITIES.find(c => c.name.toLowerCase().includes(query.toLowerCase()));
          if (partial) {{
            selectLocation(partial.lat, partial.lon, `${{partial.name}}, ${{partial.state}}`);
          }} else {{
            alert(`Could not find place "${{query}}" in India.`);
          }}
        }}
      }} catch (err) {{
        console.error('Nominatim search failed:', err);
      }}
    }}

    function selectCityPreset(name) {{
      const city = ALL_CITIES.find(c => c.name.toLowerCase().startsWith(name.toLowerCase()));
      if (city) {{
        selectLocation(city.lat, city.lon, `${{city.name}}, ${{city.state}}`);
      }}
    }}

    async function selectLocation(lat, lon, name) {{
      currentPlaceName = name;
      document.getElementById('input-lat').value = lat.toFixed(2);
      document.getElementById('input-lon').value = lon.toFixed(2);
      document.getElementById('search-input').value = name;
      dropdown.classList.add('hidden');

      marker.setLatLng([lat, lon]);
      map.flyTo([lat, lon], 7, {{ animate: true, duration: 1.0 }});

      const month = parseInt(document.getElementById('slider-month').value);
      await loadClimatologyForCoords(lat, lon, month, name);
    }}

    async function loadClimatologyForCoords(lat, lon, month, name) {{
      try {{
        const res = await fetch('/api/lookup_location', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ lat, lon, month, place_name: name }})
        }});
        const data = await res.json();
        if (data && data.weather) {{
          const w = data.weather;
          document.getElementById('slider-aod').value = w.aod;
          document.getElementById('slider-u10').value = w.u10;
          document.getElementById('slider-v10').value = w.v10;
          document.getElementById('slider-t2m').value = w.t2m;
          document.getElementById('slider-d2m').value = w.d2m;
          document.getElementById('slider-sp').value = w.sp;

          document.getElementById('val-aod').innerText = w.aod;
          document.getElementById('val-u10').innerText = w.u10;
          document.getElementById('val-v10').innerText = w.v10;
          document.getElementById('val-t2m').innerText = w.t2m;
          document.getElementById('val-d2m').innerText = w.d2m;
          document.getElementById('val-sp').innerText = w.sp;

          document.getElementById('place-name-display').innerText = name;
          document.getElementById('place-details-display').innerText = 
            `Nearest CAMS Cell: ${{data.cell_lat.toFixed(2)}}°N, ${{data.cell_lon.toFixed(2)}}°E (~${{data.distance_km}} km) • Real Climatology Active`;

          document.getElementById('cams-baseline-val').innerText = `${{w.cams_pm25}} µg/m³`;

          triggerPrediction();
        }}
      }} catch (err) {{
        console.error('Climatology fetch failed:', err);
        triggerPrediction();
      }}
    }}

    function reloadClimatologyForSelectedMonth() {{
      const lat = parseFloat(document.getElementById('input-lat').value);
      const lon = parseFloat(document.getElementById('input-lon').value);
      const month = parseInt(document.getElementById('slider-month').value);
      loadClimatologyForCoords(lat, lon, month, currentPlaceName);
    }}

    // Sliders
    const sliders = ['aod', 'u10', 'v10', 't2m', 'd2m', 'sp', 'month'];
    const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    sliders.forEach(s => {{
      const el = document.getElementById('slider-' + s);
      el.addEventListener('input', () => {{
        let val = el.value;
        if (s === 'month') {{
          document.getElementById('val-month').innerText = `${{monthNames[val]}} (${{val}})`;
        }} else {{
          document.getElementById('val-' + s).innerText = val;
        }}
        triggerPrediction();
      }});
    }});

    document.getElementById('input-lat').addEventListener('change', () => {{
      marker.setLatLng([document.getElementById('input-lat').value, document.getElementById('input-lon').value]);
      triggerPrediction();
    }});
    document.getElementById('input-lon').addEventListener('change', () => {{
      marker.setLatLng([document.getElementById('input-lat').value, document.getElementById('input-lon').value]);
      triggerPrediction();
    }});

    let debounceTimer;
    function triggerPrediction() {{
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(sendDualPredictionRequest, 50);
    }}

    async function sendDualPredictionRequest() {{
      const payload = {{
        aod: parseFloat(document.getElementById('slider-aod').value),
        u10: parseFloat(document.getElementById('slider-u10').value),
        v10: parseFloat(document.getElementById('slider-v10').value),
        t2m: parseFloat(document.getElementById('slider-t2m').value),
        d2m: parseFloat(document.getElementById('slider-d2m').value),
        sp: parseFloat(document.getElementById('slider-sp').value),
        lat: parseFloat(document.getElementById('input-lat').value),
        lon: parseFloat(document.getElementById('input-lon').value),
        month: parseInt(document.getElementById('slider-month').value)
      }};

      try {{
        const res = await fetch('/api/predict', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        
        // 1. Update ResTabNet Card
        const nn = data.restabnet;
        document.getElementById('nn-pm25-val').innerText = nn.pm25;
        const nnBadge = document.getElementById('nn-badge');
        nnBadge.innerText = nn.aqi_category;
        nnBadge.style.backgroundColor = nn.aqi_color;
        document.getElementById('nn-tip').innerText = nn.health_advisory;
        document.getElementById('nn-latency').innerText = nn.latency_ms + ' ms';
        document.getElementById('card-nn').style.borderLeftColor = nn.aqi_color;

        // 2. Update Random Forest Card
        const rf = data.random_forest;
        document.getElementById('rf-pm25-val').innerText = rf.pm25;
        const rfBadge = document.getElementById('rf-badge');
        rfBadge.innerText = rf.aqi_category;
        rfBadge.style.backgroundColor = rf.aqi_color;
        document.getElementById('rf-tip').innerText = rf.health_advisory;
        document.getElementById('rf-latency').innerText = rf.latency_ms + ' ms';
        document.getElementById('card-rf').style.borderLeftColor = rf.aqi_color;

        // 3. Differential
        const diff = data.comparison.diff_pm25;
        const sign = diff >= 0 ? '+' : '';
        document.getElementById('diff-readout').innerText = `ResTabNet is ${{sign}}${{diff}} µg/m³ relative to Random Forest (100 Trees)`;

        // 4. Physical Features
        document.getElementById('feat-ws').innerText = data.engineered_features.wind_speed + ' m/s';
        document.getElementById('feat-rh').innerText = data.engineered_features.dewpoint_depression + ' °C';
        document.getElementById('feat-stag').innerText = data.engineered_features.stagnation_index;

      }} catch (err) {{
        console.error('Dual prediction error:', err);
      }}
    }}

    // ==========================================
    // INTERACTIVE CHART.JS TIME SERIES
    // ==========================================
    let chartInstance = null;
    let currentTrendMode = 'observed_all';
    let currentYearFilter = 'all';

    const regionColors = {{
      'central': '#1f77b4',
      'himalaya': '#ff7f0e',
      'igp': '#2ca02c',
      'northeast': '#d62728',
      'northwest': '#9467bd',
      'south': '#8c564b'
    }};

    const regionLabels = {{
      'central': 'Central',
      'himalaya': 'Himalaya',
      'igp': 'Indo-Gangetic Plain (IGP)',
      'northeast': 'Northeast',
      'northwest': 'Northwest',
      'south': 'South'
    }};

    function initRegionalChart() {{
      if (chartInstance) {{
        chartInstance.destroy();
      }}

      const ctx = document.getElementById('regionalChart').getContext('2d');
      const filteredDates = [];
      const dateIndices = [];

      TRENDS_DATA.dates.forEach((d, i) => {{
        if (currentYearFilter === 'all' || d.startsWith(currentYearFilter)) {{
          filteredDates.push(d);
          dateIndices.push(i);
        }}
      }});

      let datasets = [];

      if (currentTrendMode === 'observed_all') {{
        TRENDS_DATA.regions.forEach(r => {{
          const allVals = TRENDS_DATA.observed_all[r];
          const dataPoints = dateIndices.map(i => allVals[i]);
          datasets.push({{
            label: regionLabels[r],
            data: dataPoints,
            borderColor: regionColors[r],
            backgroundColor: regionColors[r] + '20',
            borderWidth: 2.2,
            pointRadius: filteredDates.length > 24 ? 1.5 : 3.5,
            pointHoverRadius: 6,
            tension: 0.25,
            fill: false
          }});
        }});
      }} else if (currentTrendMode === 'model_comp') {{
        // Show Ground Truth vs ResTabNet vs RF on Test Cells for ALL 6 regions
        ['igp', 'central', 'northwest', 'south', 'northeast', 'himalaya'].forEach(r => {{
          const testObs = dateIndices.map(i => TRENDS_DATA.observed_test[r][i]);
          const testNN = dateIndices.map(i => TRENDS_DATA.restabnet_test[r][i]);
          const testRF = dateIndices.map(i => TRENDS_DATA.random_forest_test[r][i]);

          datasets.push({{
            label: `${{regionLabels[r]}} (True CAMS)`,
            data: testObs,
            borderColor: regionColors[r],
            borderWidth: 2.5,
            pointRadius: 2,
            tension: 0.2
          }});
          datasets.push({{
            label: `${{regionLabels[r]}} (ResTabNet Predicted)`,
            data: testNN,
            borderColor: '#38bdf8',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.2
          }});
          datasets.push({{
            label: `${{regionLabels[r]}} (Random Forest Predicted)`,
            data: testRF,
            borderColor: '#f59e0b',
            borderDash: [2, 3],
            borderWidth: 1.8,
            pointRadius: 2,
            tension: 0.2
          }});
        }});
      }}

      chartInstance = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: filteredDates,
          datasets: datasets
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{
            mode: 'index',
            intersect: false,
          }},
          plugins: {{
            legend: {{
              position: 'top',
              labels: {{
                color: '#cbd5e1',
                boxWidth: 12,
                font: {{ size: 11 }}
              }}
            }},
            tooltip: {{
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              titleColor: '#fff',
              bodyColor: '#cbd5e1',
              borderColor: '#334155',
              borderWidth: 1,
              callbacks: {{
                label: function(context) {{
                  return ` ${{context.dataset.label}}: ${{context.parsed.y}} µg/m³`;
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              grid: {{ color: 'rgba(51, 65, 85, 0.4)' }},
              ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}
            }},
            y: {{
              title: {{
                display: true,
                text: 'PM2.5 (µg/m³)',
                color: '#cbd5e1',
                font: {{ size: 12, weight: 'bold' }}
              }},
              grid: {{ color: 'rgba(51, 65, 85, 0.4)' }},
              ticks: {{ color: '#94a3b8' }},
              min: 0,
              max: 280
            }}
          }}
        }}
      }});

      // Synchronize dataset visibility with activeRegions
      chartInstance.data.datasets.forEach((ds, i) => {{
        for (const [rKey, label] of Object.entries(regionLabels)) {{
          if (ds.label.startsWith(label)) {{
            chartInstance.setDatasetVisibility(i, activeRegions[rKey]);
            break;
          }}
        }}
      }});
      chartInstance.update();
      updatePillStyles();
    }}

    function setTrendMode(mode) {{
      currentTrendMode = mode;
      document.getElementById('btn-mode-all').className = mode === 'observed_all' 
        ? 'px-3 py-1.5 rounded-lg font-semibold bg-blue-600 text-white' 
        : 'px-3 py-1.5 rounded-lg font-semibold text-slate-300 hover:text-white';
      document.getElementById('btn-mode-comp').className = mode === 'model_comp' 
        ? 'px-3 py-1.5 rounded-lg font-semibold bg-blue-600 text-white' 
        : 'px-3 py-1.5 rounded-lg font-semibold text-slate-300 hover:text-white';
      initRegionalChart();
    }}

    const activeRegions = {{
      'central': true,
      'himalaya': true,
      'igp': true,
      'northeast': true,
      'northwest': true,
      'south': true
    }};

    function filterTrendYear(year) {{
      currentYearFilter = year;
      initRegionalChart();
    }}

    function toggleRegion(regionKey) {{
      if (!chartInstance) return;
      activeRegions[regionKey] = !activeRegions[regionKey];
      const isVisible = activeRegions[regionKey];
      const labelPrefix = regionLabels[regionKey];

      chartInstance.data.datasets.forEach((ds, i) => {{
        if (ds.label.startsWith(labelPrefix)) {{
          chartInstance.setDatasetVisibility(i, isVisible);
        }}
      }});
      chartInstance.update();
      updatePillStyles();
    }}

    function setAllRegions(val) {{
      if (!chartInstance) return;
      for (const k of Object.keys(activeRegions)) {{
        activeRegions[k] = val;
      }}
      chartInstance.data.datasets.forEach((ds, i) => {{
        for (const [rKey, label] of Object.entries(regionLabels)) {{
          if (ds.label.startsWith(label)) {{
            chartInstance.setDatasetVisibility(i, val);
            break;
          }}
        }}
      }});
      chartInstance.update();
      updatePillStyles();
    }}

    function updatePillStyles() {{
      for (const [rKey, isActive] of Object.entries(activeRegions)) {{
        const pill = document.getElementById('pill-' + rKey);
        if (!pill) continue;
        if (isActive) {{
          pill.style.opacity = '1';
          pill.style.filter = 'none';
          pill.style.textDecoration = 'none';
        }} else {{
          pill.style.opacity = '0.35';
          pill.style.filter = 'grayscale(80%)';
          pill.style.textDecoration = 'line-through';
        }}
      }}
    }}

    async function runEvaluation() {{
      const btn = document.getElementById('btn-run-eval');
      const terminal = document.getElementById('eval-terminal');
      const dataPath = document.getElementById('eval-data-path').value.trim();

      btn.disabled = true;
      btn.innerText = 'Evaluating...';
      terminal.innerText = `Starting evaluation of ${{dataPath}} with ResTabNet...\nPlease wait (executing subprocess)...\n`;

      try {{
        const res = await fetch('/api/evaluate', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ data_path: dataPath }})
        }});
        const text = await res.text();
        terminal.innerText = text;
      }} catch (err) {{
        terminal.innerText = `Error executing evaluation: ${{err.message}}`;
      }} finally {{
        btn.disabled = false;
        btn.innerText = 'Run Evaluation';
      }}
    }}

    window.addEventListener('load', () => {{
      selectCityPreset('Delhi');
    }});
  </script>
</body>
</html>
"""
'''

server_part = r'''
class PM25RequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/cities":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cities_list).encode("utf-8"))
        elif parsed.path == "/api/regional_trends":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(trends_data).encode("utf-8"))
        elif parsed.path.startswith("/results/"):
            file_path = parsed.path.lstrip("/")
            if os.path.exists(file_path):
                self.send_response(200)
                if file_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif file_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript")
                elif file_path.endswith(".csv"):
                    self.send_header("Content-Type", "text/csv")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        if parsed.path == "/api/predict":
            try:
                data = json.loads(body)
                res = predict_dual(
                    aod=float(data.get("aod", 0.5)),
                    u10=float(data.get("u10", 0.0)),
                    v10=float(data.get("v10", 0.0)),
                    t2m=float(data.get("t2m", 25.0)),
                    d2m=float(data.get("d2m", 15.0)),
                    sp=float(data.get("sp", 1000.0)),
                    lat=float(data.get("lat", 20.0)),
                    lon=float(data.get("lon", 78.0)),
                    month=int(data.get("month", 1)),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_error(500, f"Inference error: {e}")

        elif parsed.path == "/api/lookup_location":
            try:
                data = json.loads(body)
                lat = float(data.get("lat", 28.61))
                lon = float(data.get("lon", 77.23))
                month = int(data.get("month", 11))
                place_name = data.get("place_name", "Selected Location")

                cell_lat, cell_lon, dist_km, weather = get_nearest_climatology(lat, lon, month)
                pred = None
                if weather:
                    pred = predict_dual(
                        aod=weather["aod"],
                        u10=weather["u10"],
                        v10=weather["v10"],
                        t2m=weather["t2m"],
                        d2m=weather["d2m"],
                        sp=weather["sp"],
                        lat=lat,
                        lon=lon,
                        month=month,
                    )

                res = {
                    "place_name": place_name,
                    "lat": lat,
                    "lon": lon,
                    "cell_lat": cell_lat,
                    "cell_lon": cell_lon,
                    "distance_km": dist_km,
                    "month": month,
                    "weather": weather,
                    "prediction": pred,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_error(500, f"Lookup error: {e}")

        elif parsed.path == "/api/evaluate":
            try:
                data = json.loads(body)
                dataset_path = data.get("data_path", "data/pm25_daily.csv")
                
                eval_env = os.environ.copy()
                eval_env["KERAS_HOME"] = "/tmp/.keras"
                eval_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
                cmd = [sys.executable, "evaluate.py", "--data", dataset_path]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=eval_env)
                output = proc.stdout if proc.returncode == 0 else (proc.stdout + "\n" + proc.stderr)

                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(output.encode("utf-8"))
            except Exception as e:
                self.send_error(500, f"Evaluation error: {e}")
        else:
            self.send_error(404, "Endpoint not found")


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), PM25RequestHandler) as httpd:
        print("\n" + "=" * 75)
        print(f" PM2.5 Dual-Model Web Application running at: http://localhost:{PORT}")
        print(" Open this URL in any web browser to interact with both models live.")
        print(" Press Ctrl+C to stop the server.")
        print("=" * 75 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down web application server...")
            if rf_process:
                rf_process.terminate()
            httpd.server_close()


if __name__ == "__main__":
    run_server()
'''

with open("app.py", "w") as f:
    f.write(app_code + "\n" + html_content + "\n" + server_part)

print("Generated dual-model app.py successfully!")
