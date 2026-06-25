# Smart Energy Monitor — single-sensor NILM

Identify which appliances are running, how many amps each draws, the units (kWh)
and the electricity bill (₹) — using **one** ZMPT101B voltage sensor and **one**
ACS712 current sensor on a shared extension board. No per-device sensor.

```
ESP32 (ZMPT101B + ACS712)
   │  every 15 s: sample 1 burst @ 4 kHz, compute 16 NILM features + V/I/P
   ▼
PC inference server (Flask)
   ├─ Random Forest (PLAID)  -> appliance TYPE  (bulb / fan / heater / microwave …)
   ├─ Step disaggregator     -> WHICH of your loads are ON + amps/kWh/₹ each
   ├─ ThingSpeak  (optional)  -> cloud history
   └─ GET /status             -> Grafana dashboard
```

## Why this design (read this first)

Your three real loads — **60 W bulb, 100 W bulb, 25 W soldering iron** — are all
**purely resistive**. Their current waveforms are identical in *shape* (clean
sine, power factor ≈ 1, no harmonics); they differ only in *magnitude*. So a
waveform classifier **cannot** tell them apart, and with one sensor you only ever
measure the *sum* of currents.

The system therefore uses a **hybrid**:

1. **Step / combinatorial disaggregator** (`python/disaggregate.py`) — the real
   workhorse for your loads. Because 25 W (0.11 A), 60 W (0.26 A) and 100 W
   (0.43 A) have distinct wattages, it finds the on/off combination whose summed
   power best matches the measured total. This correctly separates all three even
   when they are on simultaneously, and tracks amps/kWh/₹ per appliance.
2. **Random Forest on PLAID** (`python/train_plaid.py`) — the genuine "NILM
   technique" part. It classifies appliance *type* and supports future
   non-resistive loads (fan, microwave). Held-out accuracy **97.3%**
   (`models/plaid_report.txt`, `models/plaid_confusion.png`).

Both run on the same 16-feature vector defined once in `python/features.py` and
re-implemented identically on the ESP32 (Goertzel == direct DFT, matched to 1e-5).

## Quick start (no hardware needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) train the PLAID Random Forest (downloads the dataset on first run)
python python/train_plaid.py

# 2) start the inference server (tariff in ₹/kWh; ThingSpeak key optional)
python python/inference_server.py --model models/plaid_rf.joblib --tariff 8

# 3) in another terminal, simulate the ESP32 switching your 3 loads on/off
python python/simulate_esp32.py --url http://localhost:5000/ingest
```

Watch the server output / `curl localhost:5000/status` — you'll see appliances
turn on one by one with per-appliance amps, kWh and ₹.

## On real hardware

1. **Firmware:** open `firmware/nilm_monitor/nilm_monitor.ino` in Arduino IDE.
   - Set your WiFi SSID/password.
   - Set `SERVER_URL` to your PC's LAN IP, e.g. `http://192.168.1.50:5000/ingest`
     (find it with `ipconfig` / `ip addr`; PC and ESP32 must be on the same WiFi).
   - The sensor calibration (pins, ACS712 0.185 V/A, currentCalibration 0.39,
     voltageCalibration 1.75, noise subtraction) is carried over from your
     working monitor — **always boot with nothing plugged in** so it calibrates
     the noise floor, then plug loads in.
   - Flash, open Serial Monitor @115200 to confirm it connects and POSTs.
2. **Server:** run the inference server on the PC (same command as above). Add
   `--thingspeak-key YOUR_WRITE_KEY` to push to ThingSpeak.
3. **Grafana:** see `docs/GRAFANA.md`.

## Tuning the disaggregator

Edit `DEFAULT_APPLIANCES` in `python/disaggregate.py` if your measured wattages
differ from nominal (use the live amps the monitor reports for each load alone):

```python
DEFAULT_APPLIANCES = [
    Appliance("soldering_iron_25w", watts=25.0,  amps=0.11),
    Appliance("bulb_60w",           watts=60.0,  amps=0.26),
    Appliance("bulb_100w",          watts=100.0, amps=0.43),
]
```

`--tol 25` is the match tolerance in watts; lower it if loads are well separated,
raise it if readings are noisy.

## Training on your own zip / appliances

If you have a dataset zip (e.g. bulb / microwave / heater), use
`python/train_zip.py` (see its `--help`) — it extracts the same 16 features and
trains an RF in the same `joblib` format, so the server can load it with
`--model models/your_rf.joblib`. To train on data captured by *your* ESP32:
`firmware/nilm_capture/` + `python/capture_own_data.py` + `python/train_own.py`.

A model trained on the supplied `data2.zip` (bulb / heater / microwave, PLAID-style
30 kHz / 60 Hz) is included as `models/user_rf.joblib` — **98.1%** held-out
(`models/user_rf_report.txt`). Run the server with it via:

```bash
python python/train_zip.py --zip data2.zip --fs 30000 --mains 60 \
    --win 1200 --n-windows 3 --out models/user_rf.joblib   # reproduce
python python/inference_server.py --model models/user_rf.joblib --tariff 8
```

## Security

The WiFi password and ThingSpeak key you shared earlier were exposed publicly —
**regenerate the ThingSpeak Write API key and change the WiFi password.** Never
commit real keys; pass the ThingSpeak key via `--thingspeak-key` or the
`THINGSPEAK_KEY` env var.

## Layout

```
python/    features.py  train_plaid.py  disaggregate.py  inference_server.py
           simulate_esp32.py  train_zip.py  capture_own_data.py  train_own.py
firmware/  nilm_monitor/  (sends features to server)   nilm_capture/ (logs CSV)
models/    plaid_rf.joblib  plaid_report.txt  plaid_confusion.png
docs/      GRAFANA.md
```
