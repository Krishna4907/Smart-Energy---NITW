"""
Train a Random Forest on YOUR captured ESP32 features (data/own_features.csv).

This is the model to actually deploy: its features come from the same C++ code
that runs live, so there is no train/serve mismatch. Use it once you've captured
each appliance and the combinations you care about with capture_own_data.py.

  python train_own.py

Outputs models/own_rf.joblib (same bundle format as plaid_rf.joblib) plus a
confusion matrix and report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "own_features.csv"
MODELS = ROOT / "models"


def main():
    if not CSV.exists():
        print(f"No data at {CSV}. Capture some first with capture_own_data.py.")
        return
    df = pd.read_csv(CSV)
    X = df[list(FEATURE_NAMES)].to_numpy()
    y = df["label"].to_numpy()
    print(f"{len(df)} rows, classes: {sorted(set(y))}")
    for lbl in sorted(set(y)):
        print(f"  {lbl:22s} {int(np.sum(y == lbl))} rows")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n=== Held-out accuracy: {acc*100:.1f}% ===\n")
    print(classification_report(yte, pred, digits=3))

    MODELS.mkdir(exist_ok=True)
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45,
                                                           colorbar=False)
    ax.set_title(f"Own-data RF — {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(MODELS / "own_confusion.png", dpi=130)

    dump({
        "model": clf,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(clf.classes_),
        "fs": 4000.0, "mains": 50.0, "win_samples": 400,
        "source": "own ESP32 capture",
    }, MODELS / "own_rf.joblib")
    print(f"Saved -> {MODELS/'own_rf.joblib'}")


if __name__ == "__main__":
    main()
