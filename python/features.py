"""
Shared NILM feature extraction.

THIS IS THE SINGLE SOURCE OF TRUTH for the feature set. The C++ version on the
ESP32 (firmware/nilm_monitor) must compute the SAME features in the SAME order,
otherwise the model will silently misclassify.

A "sample window" is one or more whole mains cycles of the current waveform
(and optionally the voltage waveform sampled at the same instants).

Why these features (and not raw samples)?
  - The ESP32 sends a handful of numbers over WiFi instead of a whole waveform.
  - They are (mostly) independent of sample rate and absolute scaling, so a model
    trained on PLAID (30 kHz, 60 Hz) and a model trained on your own ESP32 data
    (a few kHz, 50 Hz) use the exact same columns.

Feature vector (order matters):
    0  i_rms          RMS current (A or ADC-normalised units)
    1  i_peak         peak current
    2  crest_factor   i_peak / i_rms                 (resistive ~1.41, spiky load >2)
    3  form_factor    i_rms / mean(|i|)              (pure sine ~1.11)
    4  thd            total harmonic distortion of current
    5  zero_cross     zero-crossings per cycle (normalised)
    6  h2 .. 12 h8    harmonic magnitudes 2..8 relative to fundamental (7 values)
    13 real_power     mean(v*i)        -> 0.0 if no voltage supplied
    14 apparent_power vrms * irms      -> 0.0 if no voltage supplied
    15 power_factor   real/apparent    -> 0.0 if no voltage supplied
"""

from __future__ import annotations

import numpy as np

# Harmonics 2..8 are returned relative to the fundamental (harmonic 1).
HARMONICS = list(range(2, 9))

FEATURE_NAMES = (
    ["i_rms", "i_peak", "crest_factor", "form_factor", "thd", "zero_cross"]
    + [f"h{k}" for k in HARMONICS]
    + ["real_power", "apparent_power", "power_factor"]
)
N_FEATURES = len(FEATURE_NAMES)


def _harmonic_magnitudes(sig: np.ndarray, fs: float, mains: float, n_harm: int):
    """Amplitude of harmonics 1..n_harm via a direct DFT at each exact harmonic
    frequency. This is identical to the Goertzel algorithm used on the ESP32
    (firmware), so PLAID features and on-device features stay consistent.

    Returns an array of length n_harm; index 0 is the fundamental.
    """
    n = len(sig)
    j = np.arange(n)
    mags = np.zeros(n_harm)
    nyq = fs / 2.0
    for k in range(1, n_harm + 1):
        f = k * mains
        if f >= nyq:
            break
        ang = 2.0 * np.pi * f * j / fs
        re = np.dot(sig, np.cos(ang))
        im = np.dot(sig, np.sin(ang))
        mags[k - 1] = np.sqrt(re * re + im * im) * 2.0 / n
    return mags


def extract_features(current: np.ndarray,
                     voltage: np.ndarray | None = None,
                     fs: float = 30000.0,
                     mains: float = 60.0) -> np.ndarray:
    """Compute the NILM feature vector from a current (and optional voltage) window.

    Parameters
    ----------
    current : 1-D array of current samples (any unit; DC offset removed here)
    voltage : optional 1-D array of voltage samples, same length & timing as current
    fs      : sample rate in Hz
    mains   : mains fundamental frequency (50 in India, 60 in PLAID/US)
    """
    i = np.asarray(current, dtype=np.float64)
    i = i - i.mean()                      # remove DC offset

    # Trim to a whole number of mains cycles for clean harmonics.
    spc = fs / mains
    ncyc = max(int(len(i) // spc), 1)
    keep = int(ncyc * spc)
    if keep >= 8:
        i = i[:keep]

    n = len(i)
    i_rms = float(np.sqrt(np.mean(i ** 2)))
    i_peak = float(np.max(np.abs(i)))
    mean_abs = float(np.mean(np.abs(i))) + 1e-12

    crest = i_peak / (i_rms + 1e-12)
    form = i_rms / mean_abs

    mags = _harmonic_magnitudes(i, fs, mains, max(HARMONICS))
    fund = mags[0] + 1e-12
    # THD = sqrt(sum(h_k^2 for k>=2)) / h1
    thd = float(np.sqrt(np.sum(mags[1:] ** 2)) / fund)
    harm_ratios = [float(mags[k - 1] / fund) for k in HARMONICS]

    # zero-crossings per cycle
    signs = np.sign(i)
    signs[signs == 0] = 1
    crossings = int(np.sum(signs[:-1] != signs[1:]))
    zero_cross = crossings / max(ncyc, 1)

    if voltage is not None and len(voltage) >= n and n > 0:
        v = np.asarray(voltage, dtype=np.float64)[:n]
        v = v - v.mean()
        real_power = float(np.mean(v * i))
        v_rms = float(np.sqrt(np.mean(v ** 2)))
        apparent = v_rms * i_rms
        pf = real_power / (apparent + 1e-12)
        # clamp PF to a sane range (numerical noise can push it slightly out)
        pf = float(np.clip(pf, -1.0, 1.0))
    else:
        real_power = apparent = pf = 0.0

    feats = [i_rms, i_peak, crest, form, thd, zero_cross] + harm_ratios + [
        real_power, apparent, pf
    ]
    return np.asarray(feats, dtype=np.float64)


if __name__ == "__main__":
    # quick self-test on a synthetic resistive load (pure sine current)
    fs, mains = 30000.0, 60.0
    t = np.arange(0, 0.1, 1.0 / fs)
    cur = 0.43 * np.sqrt(2) * np.sin(2 * np.pi * mains * t)
    volt = 230 * np.sqrt(2) * np.sin(2 * np.pi * mains * t)
    f = extract_features(cur, volt, fs, mains)
    for name, val in zip(FEATURE_NAMES, f):
        print(f"{name:15s} {val: .4f}")
