"""
PC / cloud inference server (option B).

The ESP32 samples one mains cycle at high rate, computes the feature vector
(same definition as python/features.py) plus the aggregate V / I / P, and POSTs
it here as JSON. This server then:

  1. Runs the Random Forest (PLAID or your own model) to classify appliance TYPE.
  2. Runs the combinatorial disaggregator to decide which of your known loads are
     ON and to split current / energy / cost per appliance.
  3. Computes total units (kWh) and the electricity bill (Rs).
  4. Optionally forwards everything to ThingSpeak.
  5. Exposes the latest result as JSON for a Grafana dashboard.

Endpoints
  POST /ingest    ESP32 posts a reading -> returns the full breakdown JSON
  GET  /status    latest breakdown (point Grafana's Infinity/JSON datasource here)
  GET  /health    liveness probe
  POST /reset     zero the per-appliance energy/cost counters

Run:
  python python/inference_server.py --model models/plaid_rf.joblib \
      --tariff 8 --thingspeak-key YOUR_WRITE_KEY
(ThingSpeak key is optional; omit to skip cloud upload.)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import requests
from flask import Flask, jsonify, request
from joblib import load

sys.path.insert(0, os.path.dirname(__file__))
from disaggregate import Appliance, Disaggregator, DEFAULT_APPLIANCES  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

app = Flask(__name__)

STATE = {"latest": {"msg": "no data yet"}, "lock": threading.Lock()}
CFG: dict = {}


def classify(feature_vector):
    """Return (label, confidence) from the loaded RF, or (None, 0) if no model."""
    bundle = CFG.get("bundle")
    if bundle is None or feature_vector is None:
        return None, 0.0
    x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != len(bundle["feature_names"]):
        return None, 0.0
    model = bundle["model"]
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return str(model.classes_[idx]), float(proba[idx])


def push_thingspeak(payload: dict):
    key = CFG.get("thingspeak_key")
    if not key:
        return
    # field1 V, field2 I, field3 P, field4 total kWh, field5 total Rs,
    # field6 #appliances on, field7 RF type confidence, field8 modelled power
    fields = {
        "field1": payload["voltage_v"],
        "field2": payload["total_current_a"],
        "field3": payload["total_power_w"],
        "field4": payload["total_energy_kwh"],
        "field5": payload["total_cost_rs"],
        "field6": len([b for b in payload["breakdown"]]),
        "field7": payload.get("rf_confidence", 0.0),
        "field8": payload["modelled_power_w"],
    }
    try:
        requests.get("https://api.thingspeak.com/update",
                     params={"api_key": key, **fields}, timeout=10)
    except requests.RequestException as e:
        print(f"[thingspeak] upload failed: {e}")


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": CFG.get("bundle") is not None})


@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json(force=True, silent=True) or {}
    voltage = float(data.get("voltage", 230.0))
    current = float(data.get("current", 0.0))
    power = float(data.get("power", voltage * current))
    feats = data.get("features")

    rf_label, rf_conf = classify(feats)
    result = CFG["disagg"].update(total_power=power, total_current=current,
                                  voltage=voltage)
    result["rf_type"] = rf_label
    result["rf_confidence"] = round(rf_conf, 3)
    result["timestamp"] = time.time()

    with STATE["lock"]:
        STATE["latest"] = result

    if CFG.get("thingspeak_key"):
        threading.Thread(target=push_thingspeak, args=(result,), daemon=True).start()

    return jsonify(result)


@app.route("/status")
def status():
    with STATE["lock"]:
        return jsonify(STATE["latest"])


@app.route("/reset", methods=["POST"])
def reset():
    CFG["disagg"].reset_energy()
    return jsonify({"ok": True, "msg": "energy/cost counters reset"})


def build_config(args):
    bundle = None
    if args.model and os.path.exists(args.model):
        bundle = load(args.model)
        print(f"Loaded model: {args.model}  classes={bundle.get('classes')}")
    else:
        print("No RF model loaded (disaggregation-only mode).")

    appliances = DEFAULT_APPLIANCES
    disagg = Disaggregator(appliances, tariff_rs_kwh=args.tariff,
                           match_tol_w=args.tol,
                           state_file=args.state_file)
    CFG.update(bundle=bundle, disagg=disagg, thingspeak_key=args.thingspeak_key)
    print(f"Appliances: {[(a.name, a.watts) for a in appliances]}")
    print(f"Tariff: Rs {args.tariff}/kWh   ThingSpeak: "
          f"{'on' if args.thingspeak_key else 'off'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/plaid_rf.joblib")
    ap.add_argument("--tariff", type=float, default=8.0, help="Rs per kWh")
    ap.add_argument("--tol", type=float, default=25.0, help="match tolerance (W)")
    ap.add_argument("--state-file", default="models/energy_state.json")
    ap.add_argument("--thingspeak-key", default=os.environ.get("THINGSPEAK_KEY"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    build_config(args)
    print(f"\nServer on http://{args.host}:{args.port}  "
          f"(ESP32 -> POST /ingest, Grafana -> GET /status)\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
