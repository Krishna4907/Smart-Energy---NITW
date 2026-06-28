"""
GRID-IQ — Live Validation Script
Tests trained model accuracy by running predictions against
known ground truth in real time. Flip appliances as prompted,
script checks predictions vs what you actually plugged in.

USAGE:
  python validate_model.py
  Follow prompts — set up each scenario, press Enter, wait 10s.
  At the end you get per-appliance accuracy to quote in your pitch.
"""

import time
import requests
import joblib
import sys
import numpy as np

# ==================== CONFIG ==========================
ESP32_IP = "10.112.255.241"
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

# Scenarios to test — covers all key cases judges will ask about
TEST_SCENARIOS = [
    {
        "name"     : "IDLE — nothing plugged in",
        "bulb100"  : 0, "bulb60": 0, "iron": 0, "charger": 0
    },
    {
        "name"     : "100W BULB only",
        "bulb100"  : 1, "bulb60": 0, "iron": 0, "charger": 0
    },
    {
        "name"     : "60W BULB only",
        "bulb100"  : 0, "bulb60": 1, "iron": 0, "charger": 0
    },
    {
        "name"     : "SOLDERING IRON only",
        "bulb100"  : 0, "bulb60": 0, "iron": 1, "charger": 0
    },
    {
        "name"     : "CHARGER only (phone at 30-50% battery)",
        "bulb100"  : 0, "bulb60": 0, "iron": 0, "charger": 1
    },
    {
        "name"     : "100W BULB + IRON",
        "bulb100"  : 1, "bulb60": 0, "iron": 1, "charger": 0
    },
    {
        "name"     : "60W BULB + CHARGER",
        "bulb100"  : 0, "bulb60": 1, "iron": 0, "charger": 1
    },
    {
        "name"     : "100W BULB + 60W BULB",
        "bulb100"  : 1, "bulb60": 1, "iron": 0, "charger": 0
    },
    {
        "name"     : "IRON + CHARGER",
        "bulb100"  : 0, "bulb60": 0, "iron": 1, "charger": 1
    },
    {
        "name"     : "ALL FOUR together",
        "bulb100"  : 1, "bulb60": 1, "iron": 1, "charger": 1
    },
]

SECONDS_PER_TEST = 10   # seconds to test each scenario


# ── Load model ────────────────────────────────────────
try:
    model  = joblib.load("grid_iq_model.joblib")
    scaler = joblib.load("grid_iq_scaler.joblib")
    print("Model loaded OK.")
except FileNotFoundError:
    print("ERROR: Model not found. Run train_model.py first.")
    sys.exit(1)


# ── Fetch from ESP32 ──────────────────────────────────
def fetch():
    try:
        r = requests.get(DATA_URL, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] {e}")
        return None


# ── Test connection first ─────────────────────────────
print("Testing ESP32 connection...")
d = fetch()
if d is None:
    print(f"ERROR: Cannot reach {DATA_URL}")
    print("Check: ESP32 powered? Same WiFi? IP correct?")
    sys.exit(1)
print(f"Connected! v_rms={d.get('v_rms')}V  i_rms={d.get('i_rms')}A\n")


# ── Tallies ───────────────────────────────────────────
# tally[label] = [correct_count, total_count]
tally       = {n: [0, 0] for n in LABELS}
mistakes    = []
total_right = 0
total_all   = 0


# ── Main validation loop ──────────────────────────────
def main():
    global total_right, total_all

    print("=" * 60)
    print("GRID-IQ Live Validation")
    print(f"Testing {len(TEST_SCENARIOS)} scenarios × {SECONDS_PER_TEST}s each")
    print("=" * 60)

    for idx, scenario in enumerate(TEST_SCENARIOS):
        ground_truth = np.array([scenario[n] for n in LABELS])

        print(f"\n[{idx+1}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        input("    Set up this scenario then press Enter...")

        start    = time.time()
        n_samples = 0
        correct_this = 0

        while time.time() - start < SECONDS_PER_TEST:
            remaining = SECONDS_PER_TEST - (time.time() - start)
            data = fetch()
            if data:
                feats = [data.get(f, 0) for f in FEATURES]
                X     = scaler.transform([feats])
                pred  = model.predict(X)[0]

                sample_correct = True
                for i, name in enumerate(LABELS):
                    tally[name][1] += 1
                    total_all      += 1
                    if pred[i] == ground_truth[i]:
                        tally[name][0] += 1
                        total_right    += 1
                    else:
                        sample_correct = False
                        mistakes.append(
                            f"{scenario['name']} | "
                            f"{NAMES[name]}: predicted={pred[i]} actual={ground_truth[i]}"
                        )

                if sample_correct:
                    correct_this += 1
                n_samples += 1

            print(
                f"  {remaining:.0f}s | "
                f"pred={[int(x) for x in pred]} "
                f"truth={list(ground_truth)} "
                f"{'OK' if sample_correct else 'WRONG'}    ",
                end="\r"
            )
            time.sleep(0.5)

        scenario_acc = (correct_this / n_samples * 100) if n_samples else 0
        print(f"\n  Scenario accuracy: {scenario_acc:.1f}%  ({correct_this}/{n_samples} samples correct)")


    # ── Final report ──────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print("\nPer-appliance accuracy:")
    for name in LABELS:
        correct, total = tally[name]
        acc = (correct / total * 100) if total else 0
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {NAMES[name]:20s} {bar} {acc:5.1f}%  ({correct}/{total})")

    overall = (total_right / total_all * 100) if total_all else 0
    print(f"\nOVERALL ACCURACY: {overall:.1f}%")
    print("(This is the number to quote in your Hitachi pitch)\n")

    if mistakes:
        print(f"Misclassifications ({len(mistakes)} total, showing first 10):")
        for m in mistakes[:10]:
            print(f"  ✗ {m}")
    else:
        print("Zero misclassifications — perfect score!")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == "__main__":
    main()