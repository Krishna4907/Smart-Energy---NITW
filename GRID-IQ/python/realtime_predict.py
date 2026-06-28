"""
GRID-IQ — Real-time Prediction + Dashboard Server v2.0
Smart Energy Monitoring System — Hitachi Energy Competition
NIT Warangal

Features:
- Per-appliance detection with confidence scores
- Energy billing in INR per appliance
- CO2 tracking with human-readable equivalents
- Monthly bill projection
- Appliance health scoring (predictive maintenance)
- Live event log (ON/OFF detection with timestamps)
- Overcurrent anomaly only (THD disabled — ESP32 ADC unreliable)
"""

import time
import threading
import requests
import joblib
import sys
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from collections import deque

# ==================== CONFIG ==========================
ESP32_IP       = "10.112.255.241"
TARIFF_INR_KWH = 7.0
CO2_KG_PER_KWH = 0.82   # India CEA 2023 grid emission factor
# ======================================================

DATA_URL = f"http://{ESP32_IP}/data"

FEATURES = [
    "v_rms", "i_rms", "p_real", "s_apparent",
    "pf", "q_reactive", "crest_factor", "thd"
]
LABELS = ["bulb100", "bulb60", "iron", "charger"]
NAMES  = {
    "bulb100" : "100W Bulb",
    "bulb60"  : "60W Bulb",
    "iron"    : "Soldering Iron",
    "charger" : "Charger"
}
ICONS = {
    "bulb100": "💡",
    "bulb60" : "💡",
    "iron"   : "🔧",
    "charger": "🔌"
}
RATED_W = {
    "bulb100": 100,
    "bulb60" :  60,
    "iron"   :  40,   # update to your iron's actual wattage
    "charger":  10
}

# CO2 equivalents for context
CAR_CO2_G_PER_KM    = 120.0   # petrol car
PHONE_CHARGE_CO2_G  =   8.5   # one phone charge
TREE_ABS_G_PER_MIN  =   0.04  # one tree per minute

# Load model
try:
    model  = joblib.load("grid_iq_model.joblib")
    scaler = joblib.load("grid_iq_scaler.joblib")
    print("Model loaded OK.")
except FileNotFoundError:
    print("ERROR: Model not found. Run train_model.py first.")
    sys.exit(1)

HAS_PROBA = hasattr(model.estimators_[0], "predict_proba")

app = Flask(__name__)
CORS(app)

# ── Baseline tracking for health scoring ──────────────
BASELINES        = {n: {"thd": None, "crest": None} for n in LABELS}
BASELINE_SAMPLES = {n: [] for n in LABELS}
BASELINE_READY   = {n: False for n in LABELS}

# ── Shared state ──────────────────────────────────────
lock  = threading.Lock()
state = {
    "live"              : {},
    "appliance_status"  : {n: 0   for n in LABELS},
    "appliance_prev"    : {n: 0   for n in LABELS},
    "appliance_energy"  : {n: 0.0 for n in LABELS},
    "appliance_cost"    : {n: 0.0 for n in LABELS},
    "appliance_co2"     : {n: 0.0 for n in LABELS},
    "appliance_conf"    : {n: 0.0 for n in LABELS},
    "appliance_health"  : {n: 100 for n in LABELS},
    "total_cost"        : 0.0,
    "total_co2_grams"   : 0.0,
    "anomaly"           : False,
    "anomaly_msg"       : "",
    "history"           : deque(maxlen=300),
    "events"            : deque(maxlen=100),
    "uptime_s"          : 0,
    "session_start"     : time.time(),
}
last_t = time.time()


# ── Confidence ────────────────────────────────────────
def get_confidence(X_scaled):
    confs = {}
    for i, name in enumerate(LABELS):
        if HAS_PROBA:
            try:
                proba = model.estimators_[i].predict_proba(X_scaled)[0]
                confs[name] = round(float(max(proba)) * 100, 1)
            except Exception:
                confs[name] = 100.0
        else:
            confs[name] = 100.0
    return confs


# ── Health scoring ────────────────────────────────────
def update_health(name, feats):
    thd   = feats[7]
    crest = feats[6]

    if not BASELINE_READY[name]:
        BASELINE_SAMPLES[name].append([thd, crest])
        if len(BASELINE_SAMPLES[name]) >= 20:
            arr = np.array(BASELINE_SAMPLES[name])
            BASELINES[name] = {
                "thd"  : float(np.mean(arr[:, 0])),
                "crest": float(np.mean(arr[:, 1])),
            }
            BASELINE_READY[name] = True
        return 100

    b = BASELINES[name]
    thd_drift   = abs(thd   - b["thd"])   / max(b["thd"],   0.01)
    crest_drift = abs(crest - b["crest"]) / max(b["crest"], 0.01)
    penalty = (thd_drift * 25) + (crest_drift * 15)
    return max(0, min(100, round(100 - penalty * 8)))


# ── Event detection ───────────────────────────────────
def detect_events(new_status, old_status, ts):
    evs = []
    for name in LABELS:
        if new_status[name] == 1 and old_status[name] == 0:
            evs.append({
                "t"     : ts,
                "name"  : NAMES[name],
                "action": "ON",
                "icon"  : ICONS[name],
            })
        elif new_status[name] == 0 and old_status[name] == 1:
            evs.append({
                "t"     : ts,
                "name"  : NAMES[name],
                "action": "OFF",
                "icon"  : ICONS[name],
            })
    return evs


# ── CO2 equivalents ───────────────────────────────────
def co2_equivalents(co2_grams):
    co2_kg = co2_grams / 1000.0
    return {
        "car_km"       : round(co2_kg * 1000 / CAR_CO2_G_PER_KM,   3),
        "phone_charges": round(co2_grams / PHONE_CHARGE_CO2_G,      2),
        "tree_minutes" : round(co2_grams / TREE_ABS_G_PER_MIN,      1),
    }


# ── Bill projection ───────────────────────────────────
def bill_projection():
    elapsed_h = (time.time() - state["session_start"]) / 3600.0
    if elapsed_h < 0.005:
        return 0.0
    total_wh   = sum(state["appliance_energy"].values())
    rate_wh_ph = total_wh / elapsed_h
    monthly_kwh= (rate_wh_ph * 24 * 30) / 1000.0
    return round(monthly_kwh * TARIFF_INR_KWH, 2)


# ── Background poll ───────────────────────────────────
def poll_loop():
    global last_t
    while True:
        try:
            r    = requests.get(DATA_URL, timeout=5)
            data = r.json()

            feats = [data.get(f, 0) for f in FEATURES]
            X     = scaler.transform([feats])
            pred  = model.predict(X)[0]
            confs = get_confidence(X)

            now  = time.time()
            dt_h = (now - last_t) / 3600.0
            last_t = now

            # Anomaly — overcurrent only, no THD
            # THD from ESP32 ADC is unreliable due to hardware non-linearity
            i_rms = data.get("i_rms", 0)
            if i_rms > 4.5:
                anom     = True
                anom_msg = f"Overcurrent: {i_rms:.2f}A"
            elif i_rms > 0.05 and all(confs[n] < 60 for n in LABELS):
                anom     = True
                anom_msg = "Unknown load detected"
            else:
                anom     = False
                anom_msg = ""

            total_on_w = sum(
                RATED_W[n] for n, on in zip(LABELS, pred) if on
            ) or 1

            with lock:
                prev = dict(state["appliance_status"])
                new  = {n: int(on) for n, on in zip(LABELS, pred)}

                evs = detect_events(new, prev, now)
                for ev in evs:
                    state["events"].appendleft(ev)

                state["live"]             = data
                state["appliance_status"] = new
                state["appliance_prev"]   = prev
                state["appliance_conf"]   = confs
                state["uptime_s"]         = data.get("uptime_s", 0)
                state["anomaly"]          = anom
                state["anomaly_msg"]      = anom_msg

                for n, on in zip(LABELS, pred):
                    if on:
                        share        = RATED_W[n] / total_on_w
                        pwr          = data.get("p_real", 0) * share
                        energy_delta = pwr * dt_h

                        state["appliance_energy"][n] += energy_delta
                        state["appliance_cost"][n]    = (
                            state["appliance_energy"][n] / 1000.0
                        ) * TARIFF_INR_KWH

                        co2_delta = (energy_delta / 1000.0) \
                                    * CO2_KG_PER_KWH * 1000
                        state["appliance_co2"][n] += co2_delta
                        state["total_co2_grams"]  += co2_delta

                        state["appliance_health"][n] = \
                            update_health(n, feats)

                state["total_cost"] = sum(
                    state["appliance_cost"].values()
                )
                state["history"].append({
                    "t": now,
                    "p": data.get("p_real", 0),
                    "i": data.get("i_rms",  0),
                    "v": data.get("v_rms",  0),
                })

        except Exception as e:
            print(f"[poll] {e}")

        time.sleep(0.5)


# ── Endpoints ─────────────────────────────────────────
@app.route("/dashboard_data")
def dashboard_data():
    with lock:
        co2 = state["total_co2_grams"]
        return jsonify({
            "live"             : state["live"],
            "appliance_status" : state["appliance_status"],
            "appliance_energy" : {
                k: round(v, 4)
                for k, v in state["appliance_energy"].items()
            },
            "appliance_cost"   : {
                k: round(v, 2)
                for k, v in state["appliance_cost"].items()
            },
            "appliance_co2"    : {
                k: round(v, 2)
                for k, v in state["appliance_co2"].items()
            },
            "appliance_conf"   : state["appliance_conf"],
            "appliance_health" : state["appliance_health"],
            "appliance_names"  : NAMES,
            "appliance_icons"  : ICONS,
            "total_cost"       : round(state["total_cost"],       2),
            "total_co2_grams"  : round(co2,                       2),
            "co2_equivalents"  : co2_equivalents(co2),
            "bill_projection"  : bill_projection(),
            "anomaly"          : state["anomaly"],
            "anomaly_msg"      : state["anomaly_msg"],
            "history"          : list(state["history"]),
            "events"           : list(state["events"])[:20],
            "uptime_s"         : state["uptime_s"],
            "baselines_ready"  : BASELINE_READY,
        })


@app.route("/reset")
def reset():
    with lock:
        for n in LABELS:
            state["appliance_energy"][n] = 0.0
            state["appliance_cost"][n]   = 0.0
            state["appliance_co2"][n]    = 0.0
        state["total_cost"]     = 0.0
        state["total_co2_grams"]= 0.0
        state["session_start"]  = time.time()
        state["events"].clear()
    return jsonify({"status": "reset ok"})


@app.route("/events")
def events():
    with lock:
        return jsonify(list(state["events"]))


if __name__ == "__main__":
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    print(f"Polling ESP32 at {DATA_URL}")
    print("Dashboard → http://localhost:5000/dashboard_data")
    print("Open dashboard/index.html in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)