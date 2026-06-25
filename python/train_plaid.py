"""
Train a Random Forest appliance classifier on the PLAID submetered dataset.

PLAID submetered format (one CSV per instance):
    column 0 = current (A), column 1 = voltage (V), 30000 Hz, 60 Hz mains, ~2 s.
    The instance id (filename without .csv) is the key into metadata_submetered.json.

We extract the shared NILM feature vector (python/features.py) from several
steady-state windows per instance, then train a RandomForest. Train/test split is
done BY INSTANCE (GroupShuffleSplit) so windows from the same recording never leak
across the split.

Outputs (models/):
    plaid_rf.joblib        trained pipeline (label encoder + RF) + metadata
    plaid_confusion.png    confusion matrix on the held-out instances
    plaid_report.txt       per-class precision / recall / F1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES, N_FEATURES, extract_features  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

FS = 30000.0          # PLAID sample rate
MAINS = 60.0          # PLAID is US 60 Hz
WIN_SAMPLES = 3000    # 0.1 s = 6 mains cycles per feature window
N_WINDOWS = 6         # windows per instance, taken from the steady-state region
STEADY_START = 24000  # skip the off->on transient at the start (~0.8 s)

# Map PLAID's fine appliance types onto the categories that matter for this
# project. Anything not listed is kept under its own name. Set --all to disable.
TARGET_MAP = {
    "Incandescent Light Bulb": "bulb",
    "Compact Fluorescent Lamp": "cfl",
    "Fan": "fan",
    "Heater": "heater",
    "Soldering Iron": "soldering_iron",
    "Hairdryer": "hairdryer",
    "Laptop": "laptop",
    "Fridge": "fridge",
    "Air Conditioner": "air_conditioner",
    "Microwave": "microwave",
    "Vacuum": "vacuum",
    "Washing Machine": "washing_machine",
}


def load_metadata():
    with open(DATA / "metadata_submetered.json") as f:
        meta = json.load(f)
    return {k: v["appliance"]["type"] for k, v in meta.items()}


def windows_from_file(path: Path):
    """Yield feature vectors from steady-state windows of one PLAID instance."""
    df = pd.read_csv(path, header=None).to_numpy()
    cur, volt = df[:, 0], df[:, 1]
    n = len(cur)
    end = n - WIN_SAMPLES
    if end <= STEADY_START:
        starts = [max(0, n - WIN_SAMPLES)]
    else:
        starts = np.linspace(STEADY_START, end, N_WINDOWS, dtype=int)
    for s in starts:
        c = cur[s:s + WIN_SAMPLES]
        v = volt[s:s + WIN_SAMPLES]
        if len(c) < WIN_SAMPLES:
            continue
        yield extract_features(c, v, FS, MAINS)


def build_dataset(keep_all: bool, limit_per_class: int | None):
    id2type = load_metadata()
    csv_dir = DATA / "submetered" / "submetered_new"
    X, y, groups = [], [], []
    class_counts: dict[str, int] = {}
    files = sorted(csv_dir.glob("*.csv"), key=lambda p: int(p.stem))
    for path in tqdm(files, desc="extracting features"):
        fid = path.stem
        ptype = id2type.get(fid)
        if ptype is None:
            continue
        label = ptype if keep_all else TARGET_MAP.get(ptype)
        if label is None:
            continue
        if limit_per_class and class_counts.get(label, 0) >= limit_per_class:
            continue
        for feats in windows_from_file(path):
            if np.all(np.isfinite(feats)):
                X.append(feats)
                y.append(label)
                groups.append(fid)
        class_counts[label] = class_counts.get(label, 0) + 1
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="keep all 16 PLAID appliance types (default: project subset)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap instances per class (debug/speed)")
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    print(f"Feature set ({N_FEATURES}): {FEATURE_NAMES}")
    X, y, groups = build_dataset(keep_all=args.all, limit_per_class=args.limit)
    print(f"\nDataset: {X.shape[0]} windows, {len(set(y))} classes")
    for lbl in sorted(set(y)):
        print(f"  {lbl:18s} {np.sum(y == lbl)} windows")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"\nTrain windows: {len(tr)}  Test windows: {len(te)} "
          f"(split by instance, no leakage)")

    clf = RandomForestClassifier(
        n_estimators=args.trees, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n=== Held-out accuracy: {acc*100:.1f}% ===\n")
    report = classification_report(yte, pred, digits=3)
    print(report)

    labels = sorted(set(y))
    (MODELS / "plaid_report.txt").write_text(
        f"PLAID Random Forest — held-out accuracy {acc*100:.1f}%\n\n{report}\n")

    cm = confusion_matrix(yte, pred, labels=labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"PLAID Random Forest — {acc*100:.1f}% accuracy")
    plt.tight_layout()
    plt.savefig(MODELS / "plaid_confusion.png", dpi=130)

    print("\nTop feature importances:")
    for idx in np.argsort(clf.feature_importances_)[::-1][:8]:
        print(f"  {FEATURE_NAMES[idx]:15s} {clf.feature_importances_[idx]:.3f}")

    dump({
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": list(clf.classes_),
        "fs": FS, "mains": MAINS,
        "win_samples": WIN_SAMPLES,
        "source": "PLAID submetered",
    }, MODELS / "plaid_rf.joblib")
    print(f"\nSaved -> {MODELS/'plaid_rf.joblib'}")
    print(f"Saved -> {MODELS/'plaid_confusion.png'}")
    print(f"Saved -> {MODELS/'plaid_report.txt'}")


if __name__ == "__main__":
    main()
