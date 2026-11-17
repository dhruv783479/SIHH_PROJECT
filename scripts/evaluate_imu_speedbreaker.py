"""Re-examine the "Marked Speed Breaker" IMU class: how much can the number be trusted?

Why this exists
---------------
The served IMU classifier reports F1 0.44 for "Marked Speed Breaker" on 164
held-out windows -- but only 11 of those windows are that class. This script
does NOT try to make the number bigger. It measures how uncertain it is and
what limits it, using only the logs already in the repository
(datasets/04_mobile_imu_telemetry_100hz/raw_logs, 10 drive CSVs).

What it computes (CPU only, a minute or two):
  1. Blocked cross-validation over ALL event windows (every drive split into
     K contiguous time blocks; fold k = block k of every drive; one window of
     guard gap removed on each side of the held-out block from training).
     Pooled out-of-fold per-class precision / recall / F1 with support.
  2. A block bootstrap (resampling whole (drive, block) units, never single
     windows) for the 95% interval of the Marked Speed Breaker F1.
  3. Leave-one-drive-out recall for every class that has more than one drive.
     "Marked Speed Breaker" currently comes from ONE drive log, so it cannot
     be tested on an unseen drive; the script says so instead of pretending.
  4. Optional new drive logs (--extra-drive NAME:CLASS:PATH) in the same CSV
     format (Time,Gx,Gy,Gz,Ax,Ay,Az). Once a second marked-speed-breaker
     drive exists, step 3 becomes possible for that class too.

Reading rule, fixed before running: an F1 for a class is described as
"unsettled" when its 95% bootstrap interval is wider than 0.30, and a class
backed by a single drive is flagged "single_drive" regardless of its F1.
Nothing here edits the served model or any repository file; results go to
the --out JSON only.

    python -m scripts.evaluate_imu_speedbreaker --out imu_speedbreaker_evaluation.json
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

CLASS_NAMES = ["Smooth Asphalt", "Unmarked Speed Breaker / Bump",
               "Marked Speed Breaker", "Pothole Impact"]
TARGET_CLASS = 2
WIDE_INTERVAL = 0.30          # reading rule: wider than this = "unsettled"

DRIVES = [  # (class id, file name) -- same mapping as scripts/fetch_real_imu_dataset.py
    (0, "plain_road_1.csv"), (0, "plain_road_2.csv"), (0, "plain_road_3.csv"),
    (0, "plain_road_4.csv"), (1, "unmarked_sb_1.csv"), (1, "unmarked_sb_2.csv"),
    (1, "unmarked_sb_3.csv"), (2, "marked_sb_1.csv"),
    (3, "potholes_1.csv"), (3, "potholes_2.csv"),
]


# ---------------------------------------------------------------- pure helpers
def block_ids(n, k):
    """Assign n time-ordered windows to k contiguous blocks (0..k-1)."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    k = max(1, min(k, n))
    return np.minimum((np.arange(n) * k) // n, k - 1).astype(int)


def guarded_train_mask(blocks, held_block, guard=1):
    """True for windows allowed in training when `held_block` is held out.
    `guard` windows next to the held-out block are dropped on both sides."""
    blocks = np.asarray(blocks)
    held = np.flatnonzero(blocks == held_block)
    mask = blocks != held_block
    if held.size:
        lo, hi = held.min(), held.max()
        mask[max(0, lo - guard):lo] = False
        mask[hi + 1:hi + 1 + guard] = False
    return mask


def prf(y_true, y_pred, cls):
    """Precision, recall, F1 and support for one class (0 when undefined)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": int(tp + fn),
            "predicted": int(tp + fp)}


def block_bootstrap_f1(y_true, y_pred, units, cls, reps=1000, seed=0):
    """95% interval of the class F1 resampling whole units (drive, block).

    Returns (low, high, n_units_with_class). Single windows are never
    resampled on their own because neighbouring windows are not independent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    units = np.asarray(units)
    uniq = np.unique(units)
    index = {u: np.flatnonzero(units == u) for u in uniq}
    n_with = sum(1 for u in uniq if np.any(y_true[index[u]] == cls))
    if n_with == 0:
        return None, None, 0
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([index[uniq[i]] for i in pick])
        if not np.any(y_true[idx] == cls):
            continue
        vals.append(prf(y_true[idx], y_pred[idx], cls)["f1"])
    if not vals:
        return None, None, n_with
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), n_with


def reading(cls_name, f1, lo, hi, n_drives, support):
    """Plain-language verdict following the rule stated in the module docstring."""
    flags = []
    if n_drives <= 1:
        flags.append("single_drive")
    if lo is None or hi is None:
        flags.append("interval_unavailable")
    elif (hi - lo) > WIDE_INTERVAL:
        flags.append("unsettled")
    if support < 30:
        flags.append("small_support")
    text = (f"{cls_name}: F1 {f1:.2f}"
            + ("" if lo is None else f" (95% block-bootstrap {lo:.2f}-{hi:.2f})")
            + f", {support} windows from {n_drives} drive log(s).")
    if flags:
        text += " Flags: " + ", ".join(flags) + "."
    return {"flags": flags, "text": text}


# ---------------------------------------------------------------- data loading
def load_event_windows(extra=None, raw_dir=None):
    """Same window construction as scripts/fetch_real_imu_dataset.py, kept per drive.

    Returns X (N,100,3), y (N,), drive (N,) strings, order (N,) window index
    within the drive. Reads only local files; never downloads."""
    from scripts.fetch_real_imu_dataset import (RAW_DIR, WINDOW_LEN, select_event_windows,
                                                slice_non_overlapping_windows)
    import csv
    raw_dir = raw_dir or RAW_DIR

    def read(path):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for r in csv.DictReader(fh):
                try:
                    ax, ay, az = float(r["Ax"]), float(r["Ay"]), float(r["Az"])
                except (KeyError, ValueError, TypeError):
                    continue
                if ax == 0.0 and ay == 0.0 and az == 0.0:
                    continue
                rows.append((ax, ay, az))
        return np.asarray(rows, dtype=np.float32)

    entries = [(c, n, os.path.join(raw_dir, n)) for c, n in DRIVES]
    for spec in (extra or []):
        name, cls, path = spec.split(":", 2)
        entries.append((int(cls), name, path))

    X, y, drive, order = [], [], [], []
    for cls, name, path in entries:
        if not os.path.exists(path):
            raise SystemExit(f"missing drive log: {path}")
        wins = select_event_windows(slice_non_overlapping_windows(read(path), WINDOW_LEN), cls)
        X.append(wins)
        y.append(np.full(len(wins), cls, dtype=np.int64))
        drive.extend([name] * len(wins))
        order.append(np.arange(len(wins)))
    return (np.concatenate(X).astype(np.float32), np.concatenate(y),
            np.asarray(drive), np.concatenate(order))


# ---------------------------------------------------------------- evaluation
def _fit_predict(Xtr, ytr, Xte, n_estimators):
    from models.imu_shock_classifier import IMUShockClassifier
    m = IMUShockClassifier(n_estimators=n_estimators).fit(Xtr, ytr)
    return m.predict(Xte)[0]


def blocked_cv(X, y, drive, order, k=5, guard=1, n_estimators=200):
    n = len(y)
    block = np.zeros(n, dtype=int)
    for d in np.unique(drive):
        idx = np.flatnonzero(drive == d)
        idx = idx[np.argsort(order[idx])]
        block[idx] = block_ids(len(idx), k)
    pred = np.full(n, -1, dtype=int)
    for fold in range(k):
        train = np.ones(n, dtype=bool)
        for d in np.unique(drive):
            idx = np.flatnonzero(drive == d)
            idx = idx[np.argsort(order[idx])]
            train[idx] = guarded_train_mask(block[idx], fold, guard)
        test = block == fold
        pred[test] = _fit_predict(X[train], y[train], X[test], n_estimators)
    units = np.array([f"{d}|{b}" for d, b in zip(drive, block)])
    return pred, units


def leave_one_drive_out(X, y, drive, n_estimators=200):
    out = {}
    for cls in range(len(CLASS_NAMES)):
        drives = sorted(set(drive[y == cls]))
        if len(drives) < 2:
            out[CLASS_NAMES[cls]] = {"status": "not_testable_single_drive",
                                     "drives": drives}
            continue
        per = {}
        for d in drives:
            test = drive == d
            pred = _fit_predict(X[~test], y[~test], X[test], n_estimators)
            per[d] = {"recall": round(float(np.mean(pred == cls)), 4),
                      "windows": int(test.sum())}
        out[CLASS_NAMES[cls]] = {"status": "tested", "per_held_out_drive": per}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="imu_speedbreaker_evaluation.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--extra-drive", action="append", default=[],
                    metavar="NAME:CLASS:PATH",
                    help="additional drive CSV (Time,Gx,Gy,Gz,Ax,Ay,Az) with its class id")
    args = ap.parse_args(argv)

    t0 = time.time()
    X, y, drive, order = load_event_windows(args.extra_drive)
    print(f"{len(y)} event windows from {len(set(drive))} drive logs")
    pred, units = blocked_cv(X, y, drive, order, k=args.folds, n_estimators=args.trees)

    classes = {}
    for cls, name in enumerate(CLASS_NAMES):
        s = prf(y, pred, cls)
        lo, hi, _ = block_bootstrap_f1(y, pred, units, cls, reps=args.reps, seed=cls)
        nd = len(set(drive[y == cls]))
        classes[name] = {**{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()},
                         "f1_ci95_block_bootstrap": [None if lo is None else round(lo, 4),
                                                     None if hi is None else round(hi, 4)],
                         "drive_logs": nd,
                         "reading": reading(name, s["f1"], lo, hi, nd, s["support"])}
        print(" ", classes[name]["reading"]["text"])

    lodo = leave_one_drive_out(X, y, drive, n_estimators=args.trees)
    report = {
        "what_this_is": "Uncertainty and limits of the Marked Speed Breaker F1, from blocked CV "
                        "over existing logs. Not a new model and not a served result.",
        "windows": int(len(y)), "drive_logs": sorted(set(drive.tolist())),
        "folds": args.folds, "guard_windows": 1, "trees": args.trees,
        "bootstrap_reps": args.reps, "interval_rule_wider_than": WIDE_INTERVAL,
        "per_class_blocked_cv": classes,
        "leave_one_drive_out": lodo,
        "label_caveat": ("Window labels come from the drive file (event windows = top 30% energy "
                         "for obstacle drives, bottom 45% for plain road), not from per-event "
                         "annotation. A class backed by one drive log can be separated from the "
                         "rest by that drive's vehicle/sensor/road alone."),
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()
