"""
Train a Random Forest on an arbitrary dataset ZIP (e.g. your bulb / microwave /
heater set). Produces the SAME joblib bundle format as train_plaid.py, so the
inference server can load it with --model.

It auto-discovers CSV files inside the zip and infers the class label from the
folder name (default) or the file name. Each CSV is read as numeric columns:
  - 2+ columns -> column 0 = current, column 1 = voltage
  - 1 column   -> current only (voltage-derived features are skipped)

Examples
  # labels come from the immediate parent folder (bulb/, microwave/, heater/)
  python train_zip.py --zip /path/to/dataset.zip --fs 30000 --mains 60

  # labels come from a filename prefix like "bulb_01.csv"
  python train_zip.py --zip data.zip --label-from filename --fs 4000 --mains 50

Inspect the zip layout first if unsure:
  python train_zip.py --zip data.zip --inspect
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES, N_FEATURES, extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def extract_zip(zip_path: Path) -> Path:
    out = DATA / (zip_path.stem + "_extracted")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    return out


def find_csvs(root: Path):
    return sorted(p for p in root.rglob("*.csv"))


def label_of(path: Path, root: Path, mode: str) -> str:
    if mode == "filename":
        stem = path.stem
        for sep in ("_", "-", " "):
            if sep in stem:
                return stem.split(sep)[0].lower()
        return stem.lower()
    # mode == "dir": the immediate parent folder is the class label
    # (robust to an extra wrapping top-level folder inside the zip)
    return path.parent.name.lower()


def windows_from_csv(path: Path, fs: float, mains: float, win: int, n_win: int):
    try:
        arr = pd.read_csv(path, header=None).to_numpy(dtype=float)
    except Exception:
        arr = pd.read_csv(path).to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cur = arr[:, 0]
    volt = arr[:, 1] if arr.shape[1] >= 2 else None
    n = len(cur)
    if n < win:
        return
    starts = np.linspace(0, n - win, n_win, dtype=int)
    for s in starts:
        c = cur[s:s + win]
        v = volt[s:s + win] if volt is not None else None
        feats = extract_features(c, v, fs, mains)
        if np.all(np.isfinite(feats)):
            yield feats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="path to the dataset .zip")
    ap.add_argument("--fs", type=float, default=30000.0, help="sample rate (Hz)")
    ap.add_argument("--mains", type=float, default=60.0, help="mains freq (Hz)")
    ap.add_argument("--label-from", choices=["dir", "filename"], default="dir")
    ap.add_argument("--win", type=int, default=None,
                    help="window length in samples (default: 0.1 s = fs/10)")
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--out", default=str(MODELS / "zip_rf.joblib"))
    ap.add_argument("--inspect", action="store_true",
                    help="just list the zip layout + inferred labels, then exit")
    args = ap.parse_args()

    win = args.win or int(args.fs / 10)
    zip_path = Path(args.zip).expanduser()
    if not zip_path.exists():
        sys.exit(f"zip not found: {zip_path}")

    root = extract_zip(zip_path)
    csvs = find_csvs(root)
    if not csvs:
        sys.exit(f"no CSV files found inside {zip_path}")

    if args.inspect:
        print(f"{len(csvs)} CSV files. Sample labels (--label-from {args.label_from}):")
        seen: dict[str, int] = {}
        for p in csvs:
            lbl = label_of(p, root, args.label_from)
            seen[lbl] = seen.get(lbl, 0) + 1
        for lbl, n in sorted(seen.items()):
            print(f"  {lbl:20s} {n} files")
        print("\nFirst few rows of one CSV:")
        print(pd.read_csv(csvs[0], header=None).head())
        return

    print(f"Feature set ({N_FEATURES}): {FEATURE_NAMES}")
    X, y, groups = [], [], []
    for gi, path in enumerate(csvs):
        lbl = label_of(path, root, args.label_from)
        for feats in windows_from_csv(path, args.fs, args.mains, win, args.n_windows):
            X.append(feats)
            y.append(lbl)
            groups.append(gi)
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    print(f"\nDataset: {X.shape[0]} windows, {len(set(y))} classes")
    for lbl in sorted(set(y)):
        print(f"  {lbl:20s} {int(np.sum(y == lbl))} windows")

    if len(set(y)) < 2:
        sys.exit("need at least 2 classes to train a classifier")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    clf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    acc = accuracy_score(y[te], pred)
    print(f"\n=== Held-out accuracy: {acc*100:.1f}% ===\n")
    print(classification_report(y[te], pred, digits=3))

    labels = sorted(set(y))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y[te], pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"ZIP Random Forest — {acc*100:.1f}% accuracy")
    plt.tight_layout()
    png = Path(args.out).with_suffix(".png")
    plt.savefig(png, dpi=130)

    MODELS.mkdir(exist_ok=True)
    dump({
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": list(clf.classes_),
        "fs": args.fs, "mains": args.mains, "win_samples": win,
        "source": zip_path.name,
    }, args.out)
    print(f"Saved -> {args.out}")
    print(f"Saved -> {png}")


if __name__ == "__main__":
    main()
