"""
Simulate the ESP32 so you can test the server + Grafana with NO hardware.

It generates realistic current/voltage waveforms for your three resistive loads
(switching combinations over time), extracts the SAME features, and POSTs them to
the inference server exactly like the firmware would.

  python simulate_esp32.py --url http://localhost:5000/ingest
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features  # noqa: E402

FS = 4000.0
MAINS = 50.0
N = 400
VRMS = 230.0

# (label, watts) -> resistive current amplitude derived from watts
LOADS = {
    "soldering_iron_25w": 25.0,
    "bulb_60w": 60.0,
    "bulb_100w": 100.0,
}

# scripted on/off scenario: list of sets of active loads, each held ~15s
SCENARIO = [
    set(),
    {"bulb_100w"},
    {"bulb_100w", "bulb_60w"},
    {"bulb_100w", "bulb_60w", "soldering_iron_25w"},
    {"bulb_60w", "soldering_iron_25w"},
    {"soldering_iron_25w"},
    set(),
]


def make_window(active: set[str]):
    t = np.arange(N) / FS
    v = VRMS * np.sqrt(2) * np.sin(2 * np.pi * MAINS * t)
    i = np.zeros(N)
    for name in active:
        watts = LOADS[name]
        amp = (watts / VRMS) * np.sqrt(2)             # resistive, in phase
        i += amp * np.sin(2 * np.pi * MAINS * t)
    # add a little ADC-like noise
    i += np.random.normal(0, 0.01, N)
    return i, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5000/ingest")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between posts")
    ap.add_argument("--hold", type=float, default=15.0, help="seconds per scenario step")
    args = ap.parse_args()

    print("Simulating ESP32 -> server. Ctrl-C to stop.")
    step = 0
    step_started = time.time()
    while True:
        active = SCENARIO[step % len(SCENARIO)]
        if time.time() - step_started > args.hold:
            step += 1
            step_started = time.time()
            active = SCENARIO[step % len(SCENARIO)]
            print(f"\n--- now ON: {sorted(active) or ['nothing']} ---")

        i, v = make_window(active)
        feats = extract_features(i, v, FS, MAINS)
        irms = float(np.sqrt(np.mean(i ** 2)))
        power = float(np.mean(v * i))
        payload = {"voltage": VRMS, "current": round(irms, 3),
                   "power": round(max(power, 0), 1), "features": feats.tolist()}
        try:
            r = requests.post(args.url, json=payload, timeout=5).json()
            print(f"P={payload['power']:6.1f}W -> {r['active']}  "
                  f"kWh={r['total_energy_kwh']:.4f}  Rs={r['total_cost_rs']:.3f}")
        except requests.RequestException as e:
            print(f"post failed: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
