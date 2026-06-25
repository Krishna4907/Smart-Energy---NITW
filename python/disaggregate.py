"""
Single-sensor load disaggregation for a small set of known appliances.

Why this module exists
----------------------
Your three loads (60W bulb, 100W bulb, 25W soldering iron) are all *resistive*, so
their current WAVEFORMS are identical in shape -- a Random Forest cannot tell them
apart. What DOES separate them is their wattage. With one sensor you only measure
the TOTAL power, so we recover "who is on" by finding the combination of known
appliances whose wattages best add up to the measured total.

With N appliances there are 2^N on/off combinations (8 for three appliances). We
pick the combination whose summed nominal power is closest to the measured power.
This is the standard "optimisation / combinatorial" approach to NILM for a known,
small appliance set, and it handles several appliances being ON at the same time.

It also keeps a running per-appliance energy (kWh) and cost (Rs) total.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Appliance:
    name: str
    watts: float          # nominal real power when ON
    amps: float = 0.0     # nominal RMS current when ON (at ~230 V)


@dataclass
class ApplianceState:
    name: str
    watts: float
    amps: float
    on: bool = False
    energy_kwh: float = 0.0
    cost_rs: float = 0.0


class Disaggregator:
    """Combinatorial single-sensor disaggregator + energy/cost accumulator.

    Parameters
    ----------
    appliances     : list[Appliance] with nominal wattage of each known load
    tariff_rs_kwh  : electricity tariff in Rs per kWh (India domestic slab)
    match_tol_w    : how far the best combo may be from measured power and still
                     be accepted (absolute watts); also covers a small unknown base load
    state_file     : optional JSON to persist energy totals across restarts
    """

    def __init__(self, appliances: list[Appliance], tariff_rs_kwh: float = 8.0,
                 match_tol_w: float = 25.0, state_file: str | None = None):
        self.appliances = appliances
        self.tariff = tariff_rs_kwh
        self.match_tol_w = match_tol_w
        self.state_file = Path(state_file) if state_file else None
        self.states = {a.name: ApplianceState(a.name, a.watts, a.amps)
                       for a in appliances}
        self._last_t: float | None = None
        # precompute every on/off combination once
        self._combos = list(itertools.product([0, 1], repeat=len(appliances)))
        if self.state_file and self.state_file.exists():
            self._load()

    # ----- core disaggregation -------------------------------------------
    def best_combination(self, total_power: float):
        """Return (active_names, modelled_power, residual) best matching total_power."""
        best = None
        for combo in self._combos:
            modelled = sum(a.watts for a, on in zip(self.appliances, combo) if on)
            residual = abs(total_power - modelled)
            if best is None or residual < best[2]:
                active = [a.name for a, on in zip(self.appliances, combo) if on]
                best = (active, modelled, residual)
        return best

    def update(self, total_power: float, total_current: float = 0.0,
               voltage: float = 230.0, now: float | None = None) -> dict:
        """Feed one aggregate reading; returns the per-appliance breakdown."""
        now = now if now is not None else time.time()
        active, modelled, residual = self.best_combination(total_power)

        # If the residual is large the combo is unreliable (an unknown load, or a
        # transient). Treat large positive residual as an "unknown" appliance.
        reliable = residual <= self.match_tol_w + 0.15 * max(total_power, 1.0)

        # accumulate energy for whatever is ON since the last reading
        dt_h = 0.0
        if self._last_t is not None:
            dt_h = (now - self._last_t) / 3600.0
        self._last_t = now

        for name, st in self.states.items():
            st.on = reliable and (name in active)
            if st.on and dt_h > 0:
                e = st.watts * dt_h / 1000.0
                st.energy_kwh += e
                st.cost_rs += e * self.tariff

        # split measured current among active loads in proportion to nominal amps
        breakdown = []
        amp_sum = sum(self.states[n].amps for n in active) or 1.0
        for name in active:
            st = self.states[name]
            share = (st.amps / amp_sum) * total_current if total_current else st.amps
            breakdown.append({
                "name": name, "watts": st.watts,
                "amps": round(share, 3),
                "energy_kwh": round(st.energy_kwh, 5),
                "cost_rs": round(st.cost_rs, 3),
            })

        result = {
            "active": active if reliable else ["unknown"],
            "reliable": reliable,
            "total_power_w": round(total_power, 1),
            "modelled_power_w": round(modelled, 1),
            "residual_w": round(residual, 1),
            "total_current_a": round(total_current, 3),
            "voltage_v": round(voltage, 1),
            "tariff_rs_per_kwh": self.tariff,
            "breakdown": breakdown,
            "total_energy_kwh": round(sum(s.energy_kwh for s in self.states.values()), 5),
            "total_cost_rs": round(sum(s.cost_rs for s in self.states.values()), 3),
        }
        if self.state_file:
            self._save()
        return result

    # ----- persistence ----------------------------------------------------
    def _save(self):
        self.state_file.write_text(json.dumps(
            {n: asdict(s) for n, s in self.states.items()}, indent=2))

    def _load(self):
        data = json.loads(self.state_file.read_text())
        for n, s in data.items():
            if n in self.states:
                self.states[n].energy_kwh = s.get("energy_kwh", 0.0)
                self.states[n].cost_rs = s.get("cost_rs", 0.0)

    def reset_energy(self):
        for s in self.states.values():
            s.energy_kwh = 0.0
            s.cost_rs = 0.0
        if self.state_file:
            self._save()


# Default config for THIS project (edit wattages to your measured values).
DEFAULT_APPLIANCES = [
    Appliance("soldering_iron_25w", watts=25.0, amps=0.11),
    Appliance("bulb_60w",          watts=60.0, amps=0.26),
    Appliance("bulb_100w",         watts=100.0, amps=0.43),
]


if __name__ == "__main__":
    d = Disaggregator(DEFAULT_APPLIANCES, tariff_rs_kwh=8.0)
    print("Self-test: feed several aggregate power levels\n")
    for p in [0, 25, 60, 100, 85, 160, 185, 125]:
        r = d.update(total_power=p, total_current=p / 230.0, now=time.time())
        print(f"measured {p:4d} W -> ON: {r['active']}  (modelled {r['modelled_power_w']} W, "
              f"residual {r['residual_w']} W)")
