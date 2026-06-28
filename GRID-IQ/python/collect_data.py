"""
GRID-IQ — Data Collection Script
Polls ESP32 live and saves labeled rows to CSV for model training.

USAGE:
  1. Flash firmware, confirm http://ESP32_IP/data shows live JSON.
  2. Activate venv:  venv/Scripts/activate
  3. Run:  python collect_data.py
  4. Follow prompts exactly — plug ONLY what each step says, press Enter.
"""

import requests, time, csv, os, sys

ESP32_IP      = "10.112.255.241"
DATA_URL      = f"http://10.112.255.241/data"
OUTPUT_CSV    = "grid_iq_dataset.csv"
SECONDS_PER   = 90    # seconds to log per scenario (more = better model)
POLL_INTERVAL = 0.25  # seconds between readings (~4 samples/sec)

FEATURES = ["v_rms","i_rms","p_real","s_apparent","pf",
            "q_reactive","crest_factor","thd"]
LABELS   = ["label_bulb100","label_bulb60","label_iron","label_charger"]

SCENARIOS = [
    # single appliances
    {"name":"IDLE — nothing plugged in",
     "label_bulb100":0,"label_bulb60":0,"label_iron":0,"label_charger":0},
    {"name":"100W BULB only",
     "label_bulb100":1,"label_bulb60":0,"label_iron":0,"label_charger":0},
    {"name":"60W BULB only",
     "label_bulb100":0,"label_bulb60":1,"label_iron":0,"label_charger":0},
    {"name":"SOLDERING IRON only",
     "label_bulb100":0,"label_bulb60":0,"label_iron":1,"label_charger":0},
    {"name":"CHARGER only (phone or laptop)",
     "label_bulb100":0,"label_bulb60":0,"label_iron":0,"label_charger":1},
    # two appliances
    {"name":"100W BULB + 60W BULB",
     "label_bulb100":1,"label_bulb60":1,"label_iron":0,"label_charger":0},
    {"name":"100W BULB + IRON",
     "label_bulb100":1,"label_bulb60":0,"label_iron":1,"label_charger":0},
    {"name":"100W BULB + CHARGER",
     "label_bulb100":1,"label_bulb60":0,"label_iron":0,"label_charger":1},
    {"name":"60W BULB + IRON",
     "label_bulb100":0,"label_bulb60":1,"label_iron":1,"label_charger":0},
    {"name":"60W BULB + CHARGER",
     "label_bulb100":0,"label_bulb60":1,"label_iron":0,"label_charger":1},
    {"name":"IRON + CHARGER",
     "label_bulb100":0,"label_bulb60":0,"label_iron":1,"label_charger":1},
    # three appliances
    {"name":"100W BULB + 60W BULB + IRON",
     "label_bulb100":1,"label_bulb60":1,"label_iron":1,"label_charger":0},
    {"name":"100W BULB + 60W BULB + CHARGER",
     "label_bulb100":1,"label_bulb60":1,"label_iron":0,"label_charger":1},
    {"name":"100W BULB + IRON + CHARGER",
     "label_bulb100":1,"label_bulb60":0,"label_iron":1,"label_charger":1},
    {"name":"60W BULB + IRON + CHARGER",
     "label_bulb100":0,"label_bulb60":1,"label_iron":1,"label_charger":1},
    # all four
    {"name":"ALL FOUR — 100W + 60W + IRON + CHARGER",
     "label_bulb100":1,"label_bulb60":1,"label_iron":1,"label_charger":1},
]


def fetch():
    try:
        r = requests.get(DATA_URL, timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] {e}")
        return None


def main():
    # Test connection first
    print("Testing connection to ESP32...")
    d = fetch()
    if d is None:
        print(f"\nERROR: Cannot reach {DATA_URL}")
        print("Check: ESP32 powered? Same WiFi? IP correct?")
        sys.exit(1)
    print(f"Connected! v_rms={d.get('v_rms')}V  i_rms={d.get('i_rms')}A")
    print(f"Logging to {OUTPUT_CSV}  ({SECONDS_PER}s per scenario)\n")

    file_exists = os.path.isfile(OUTPUT_CSV)
    f = open(OUTPUT_CSV, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FEATURES + LABELS + ["timestamp"])
    if not file_exists:
        writer.writeheader()

    total_rows = 0
    for idx, scenario in enumerate(SCENARIOS):
        label_vals = {k: scenario[k] for k in LABELS}
        print(f"[{idx+1}/{len(SCENARIOS)}] Set up: {scenario['name']}")
        input("    Press Enter when ready...")
        start = time.time()
        count = 0
        last_d = {"v_rms": 0, "i_rms": 0, "p_real": 0}  # safe fallback for display
        while time.time() - start < SECONDS_PER:
            remaining = SECONDS_PER - (time.time() - start)
            d = fetch()
            if d:
                last_d = d
                row = {feat: d.get(feat, 0) for feat in FEATURES}
                row.update(label_vals)
                row["timestamp"] = time.time()
                writer.writerow(row)
                count += 1
            print(f"  {remaining:.0f}s left | "
                  f"V={last_d.get('v_rms',0):.1f}V "
                  f"I={last_d.get('i_rms',0):.3f}A "
                  f"P={last_d.get('p_real',0):.1f}W    ", end="\r")
            time.sleep(POLL_INTERVAL)
        f.flush()
        total_rows += count
        print(f"\n  Saved {count} rows for '{scenario['name']}'")

    f.close()
    print(f"\n=== DONE. Total rows: {total_rows} → {OUTPUT_CSV} ===")
    print("Next step: python train_model.py")

if __name__ == "__main__":
    main()