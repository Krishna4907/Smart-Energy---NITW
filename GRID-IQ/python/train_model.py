"""
GRID-IQ — Model Training Script
Loads grid_iq_dataset.csv, trains kNN / RandomForest / MLP as multi-label
classifiers, evaluates all three, prints full report, saves best model.

USAGE:
  python train_model.py
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import joblib

DATA_CSV   = "grid_iq_dataset.csv"
MODEL_OUT  = "grid_iq_model.joblib"
SCALER_OUT = "grid_iq_scaler.joblib"

FEATURES = [
    "v_rms", "i_rms", "p_real", "s_apparent",
    "pf", "q_reactive", "crest_factor", "thd"
]
LABELS = [
    "label_bulb100", "label_bulb60",
    "label_iron", "label_charger"
]
NAMES = ["100W Bulb", "60W Bulb", "Soldering Iron", "Charger"]


def main():
    # ── Load ──────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_CSV).dropna()
    except FileNotFoundError:
        print(f"ERROR: {DATA_CSV} not found. Run collect_data.py first.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {DATA_CSV}")
    print(f"\nLabel distribution:")
    for lbl, name in zip(LABELS, NAMES):
        on  = int(df[lbl].sum())
        off = len(df) - on
        print(f"  {name:20s} → ON={on}  OFF={off}")

    if len(df) < 50:
        print("\nWARNING: Very few rows. Collect more data for reliable model.")

    X = df[FEATURES].values.astype(float)
    y = df[LABELS].values.astype(int)

    # ── Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ── Scale ─────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Models ────────────────────────────────────────
    candidates = {
        "kNN (k=5)": MultiOutputClassifier(
            KNeighborsClassifier(n_neighbors=5)
        ),
        "RandomForest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
        ),
        "MLP": MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=3000,
                early_stopping=True,
                random_state=42
            )
        ),
    }

    # ── Train + Evaluate ──────────────────────────────
    results = {}
    for name, model in candidates.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        model.fit(X_train_s, y_train)
        preds   = model.predict(X_test_s)
        exact   = accuracy_score(y_test, preds)
        hamming = hamming_loss(y_test, preds)
        results[name] = (model, exact, hamming)
        print(f"  Exact-match accuracy : {exact*100:.1f}%")
        print(f"  Hamming loss         : {hamming:.4f}  (lower = better)")
        print(f"\n  Per-appliance breakdown:")
        print(
            classification_report(
                y_test, preds,
                target_names=NAMES,
                zero_division=0
            )
        )

    # ── Pick best ─────────────────────────────────────
    best_name, (best_model, best_acc, best_hl) = max(
        results.items(), key=lambda x: x[1][1]
    )

    print("=" * 60)
    print(f"\nBEST MODEL  : {best_name}")
    print(f"Accuracy    : {best_acc*100:.1f}%")
    print(f"Hamming loss: {best_hl:.4f}")

    # ── Save ──────────────────────────────────────────
    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(scaler,     SCALER_OUT)
    print(f"\nSaved model  → {MODEL_OUT}")
    print(f"Saved scaler → {SCALER_OUT}")
    print("\nNext step: python realtime_predict.py")


if __name__ == "__main__":
    main()