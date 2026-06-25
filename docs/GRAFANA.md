# Grafana dashboard

You have two ways to feed Grafana. Pick one (or use both).

## Option A — straight from the inference server (recommended, real-time)

The server exposes the latest breakdown at `GET http://<PC-IP>:5000/status` as JSON.
Use the **Infinity** datasource to read it.

1. In Grafana: **Connections → Add new connection → Infinity** (install
   `yesoreyeram-infinity-datasource` if not present), then **Add datasource**.
   No auth needed for a LAN server.
2. **Dashboards → Import** and upload `docs/grafana_dashboard.json`.
3. When prompted, select your Infinity datasource. If your server isn't on
   `localhost:5000`, edit each panel's URL to `http://<PC-IP>:5000/status`.

The dashboard has:
- **Total power (W)**, **Total current (A)**, **Voltage (V)** — stat panels
- **Energy today (kWh)** and **Bill (₹)** — stat panels
- **Per-appliance breakdown** — table: name, watts, amps, kWh, ₹
- **Appliances ON** — count

Set the dashboard auto-refresh to 15 s (top-right) to match the ESP32 cadence.

### JSON fields available at /status
```
total_power_w, total_current_a, voltage_v,
total_energy_kwh, total_cost_rs, modelled_power_w, residual_w,
active: [names], reliable, rf_type, rf_confidence,
breakdown: [ { name, watts, amps, energy_kwh, cost_rs }, ... ]
```

## Option B — from ThingSpeak (cloud history)

Run the server with `--thingspeak-key YOUR_WRITE_KEY`. It writes:
```
field1 Voltage (V)        field5 Total cost (₹)
field2 Current (A)        field6 # appliances ON
field3 Power (W)          field7 RF confidence
field4 Total energy (kWh) field8 Modelled power (W)
```
Then either:
- Use Grafana's **JSON/Infinity** datasource against the ThingSpeak read API:
  `https://api.thingspeak.com/channels/<CHANNEL_ID>/feeds.json?api_key=<READ_KEY>&results=100`
  (parse `feeds[*].field1..field8`), or
- Just use ThingSpeak's own built-in charts.

> Per-appliance breakdown (table) is only available via Option A, because
> ThingSpeak's 8 fields hold the aggregate totals, not the variable-length list.
