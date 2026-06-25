"""
Log labelled NILM feature rows from the ESP32 (firmware/nilm_capture) over serial.

Usage:
  python capture_own_data.py --port /dev/ttyUSB0 --label bulb_100w --seconds 60
  python capture_own_data.py --port COM5 --label all_three --seconds 60

Plug in the load(s) that match --label, then run the command. Rows are appended
to data/own_features.csv. Repeat for every appliance and every combination you
want the model to recognise (don't forget an "off" label with nothing plugged in).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "own_features.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--label", required=True, help="appliance/combination label")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    new_file = not OUT.exists()
    OUT.parent.mkdir(exist_ok=True)
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(2)  # let the board reset

    n = 0
    t0 = time.time()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label"] + list(FEATURE_NAMES))
        print(f"Recording label='{args.label}' for {args.seconds}s ... (Ctrl-C to stop)")
        try:
            while time.time() - t0 < args.seconds:
                line = ser.readline().decode(errors="ignore").strip()
                if not line.startswith("FEAT,"):
                    continue
                parts = line.split(",")[1:]
                if len(parts) != len(FEATURE_NAMES):
                    continue
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue
                w.writerow([args.label] + vals)
                n += 1
                if n % 10 == 0:
                    print(f"  {n} rows")
        except KeyboardInterrupt:
            pass
    ser.close()
    print(f"Saved {n} rows for '{args.label}' -> {OUT}")


if __name__ == "__main__":
    main()
