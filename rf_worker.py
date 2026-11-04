import sys
import json
import math
import warnings
import joblib
import numpy as np

warnings.filterwarnings('ignore')

# Load models once at startup
rf = joblib.load('results/step4/random_forest.joblib')
hgb = joblib.load('results/step4/hgb_model.joblib')
advectsr_data = joblib.load('results/step4/advectsr_model.joblib')
advectsr_mean = advectsr_data['mean_model']
advectsr_unc = advectsr_data['unc_model']

sys.stdout.write("READY\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "PING":
        sys.stdout.write("PONG\n")
        sys.stdout.flush()
        continue
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            aod = float(data["aod"])
            u10 = float(data["u10"])
            v10 = float(data["v10"])
            t2m = float(data["t2m"])
            d2m = float(data["d2m"])
            sp = float(data["sp"])
            lat = float(data["lat"])
            lon = float(data["lon"])
            month = float(data["month"])
        elif isinstance(data, list):
            # [aod, u10, v10, t2m, d2m, sp, lat, lon, month]
            aod, u10, v10, t2m, d2m, sp, lat, lon, month = [float(x) for x in data[:9]]
        else:
            raise ValueError("Unsupported input format")

        # Temperature / pressure normalization units check
        t2m_k = t2m if t2m > 150 else t2m + 273.15
        d2m_k = d2m if d2m > 150 else d2m + 273.15
        sp_pa = sp if sp > 10000 else sp * 100.0

        # RF 9 features
        rf_feats = np.array([[aod, u10, v10, t2m_k, d2m_k, sp_pa, lat, lon, month]], dtype=np.float32)
        rf_pred = float(rf.predict(rf_feats)[0])

        # HGB 13 features
        ws = math.sqrt(u10**2 + v10**2)
        rh_proxy = t2m - d2m
        sin_mon = math.sin(2.0 * math.pi * month / 12.0)
        cos_mon = math.cos(2.0 * math.pi * month / 12.0)
        aod_vent = aod / (ws + 0.5)

        hgb_feats = np.array([[aod, u10, v10, ws, t2m_k, d2m_k, rh_proxy, sp_pa, lat, lon, sin_mon, cos_mon, aod_vent]], dtype=np.float32)
        hgb_pred = float(hgb.predict(hgb_feats)[0])

        # AdvectSR-PM 18 physical advection features
        u_dir = u10 / (ws + 0.1)
        v_dir = v10 / (ws + 0.1)
        adv_flux = aod * ws
        theta = math.radians(30.0)
        wind_norm = -u10 * math.sin(theta) + v10 * math.cos(theta)
        wind_para = u10 * math.cos(theta) + v10 * math.sin(theta)

        adv_feats = np.array([[
            aod, u10, v10, ws, u_dir, v_dir, adv_flux, aod_vent,
            t2m_k, d2m_k, rh_proxy, sp_pa, lat, lon, sin_mon, cos_mon,
            wind_norm, wind_para
        ]], dtype=np.float32)

        adv_pred = float(np.clip(advectsr_mean.predict(adv_feats)[0], 0, None))
        adv_unc_val = float(np.clip(advectsr_unc.predict(adv_feats)[0], 0.5, None))

        resp = json.dumps({
            "rf": round(rf_pred, 2),
            "hgb": round(hgb_pred, 2),
            "advectsr": round(adv_pred, 2),
            "advectsr_unc": round(adv_unc_val, 2)
        })
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"ERROR: {e}\n")
        sys.stdout.flush()
