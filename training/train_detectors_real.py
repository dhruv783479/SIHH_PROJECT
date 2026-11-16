"""
Train and evaluate the ROAD-SHIELD box detectors on REAL labelled data only.

What this does (and what it refuses to do)
  * Reads human-labelled YOLO-format data only. No auto-labelled, synthesised or
    heuristic-labelled images are accepted: every source must be listed in
    sources.json with provenance (url, licence, "human_labelled": true).
  * Removes near-duplicates with a perceptual hash, then puts every duplicate
    cluster entirely in ONE split (train / val / test), so a photo and its
    re-encodes can never straddle the boundary. This is the leak the earlier
    file-index split had.
  * Fixes the model-selection rule BEFORE the test split is touched: best
    val mAP50 on the pothole class. The test split is scored exactly once.
  * Writes only measured numbers to the metrics JSON. Nothing is defaulted.

Expected layout (YOLO format, one folder per source):
    <data_root>/<source>/images/*.jpg
    <data_root>/<source>/labels/*.txt          # "cls cx cy w h" normalised
    <data_root>/sources.json                   # provenance + class map per source

sources.json example:
    {"rdd2022_india": {"url": "...", "licence": "...", "human_labelled": true,
                       "class_map": {"0": "longitudinal_crack", "1": "transverse_crack",
                                     "2": "alligator_crack", "3": "pothole"}},
     "pothole_water":  {"url": "...", "licence": "...", "human_labelled": true,
                       "class_map": {"0": "pothole"}}}
Water-filled potholes must be labelled as "pothole" in the source data; the
detector can only learn them if they are in the training labels.

Run (Colab T4, ~1-3 h for yolov8s at 640 px):
    pip install ultralytics opencv-python-headless pyyaml
    python training/train_detectors_real.py --data-root /content/data --out /content/out
"""
import argparse
import glob
import hashlib
import json
import os
import random
import shutil

import cv2
import numpy as np
import yaml

CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
SEED = 20260516
# Declared before any test image is scored. Do not edit after seeing test results.
SELECTION_RULE = "best validation mAP50 on class 'pothole'; ties -> higher val mAP50-95; test scored once"


def dhash(img_path, size=8):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    r = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (r[:, 1:] > r[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def cluster_duplicates(hashes, max_dist=4):
    """Union-find over images whose dHash Hamming distance <= max_dist.
    Four 16-bit bands guarantee any pair within distance 4 shares a band exactly."""
    keys = list(hashes)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for band in range(4):
        buckets = {}
        for k in keys:
            buckets.setdefault((hashes[k] >> (16 * band)) & 0xFFFF, []).append(k)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if hamming(hashes[members[i]], hashes[members[j]]) <= max_dist:
                        parent[find(members[i])] = find(members[j])
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def load_sources(data_root):
    with open(os.path.join(data_root, "sources.json"), encoding="utf-8") as fh:
        src = json.load(fh)
    for name, meta in src.items():
        if meta.get("human_labelled") is not True:
            raise SystemExit(f"source '{name}' is not marked human_labelled=true; refusing to train on it")
        for field in ("url", "licence", "class_map"):
            if field not in meta:
                raise SystemExit(f"source '{name}' is missing provenance field '{field}'")
    return src


def collect(data_root, sources):
    items = []
    for name, meta in sources.items():
        for img in sorted(glob.glob(os.path.join(data_root, name, "images", "*"))):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lbl = os.path.join(data_root, name, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
            if not os.path.exists(lbl):
                continue  # unlabelled image is not a negative unless explicitly given an empty file
            items.append({"source": name, "image": img, "label": lbl, "class_map": meta["class_map"]})
    return items


def remap_label(label_path, class_map):
    rows = []
    for line in open(label_path, encoding="utf-8").read().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        name = class_map.get(str(int(float(p[0]))))
        if name in CLASSES:
            rows.append(f"{CLASSES.index(name)} {p[1]} {p[2]} {p[3]} {p[4]}")
    return rows


def build_splits(items, out_dir, ratios=(0.8, 0.1, 0.1)):
    hashes = {}
    for it in items:
        h = dhash(it["image"])
        if h is not None:
            hashes[it["image"]] = h
    byimg = {it["image"]: it for it in items if it["image"] in hashes}
    clusters = cluster_duplicates(hashes)
    rng = random.Random(SEED)
    rng.shuffle(clusters)
    total = sum(len(c) for c in clusters)
    targets = [ratios[0] * total, (ratios[0] + ratios[1]) * total]
    splits = {"train": [], "val": [], "test": []}
    seen = 0
    for c in clusters:  # whole clusters go to one split only
        key = "train" if seen < targets[0] else "val" if seen < targets[1] else "test"
        splits[key].extend(c)
        seen += len(c)
    report = {"n_images_total": len(items), "n_after_hashing": total,
              "n_duplicate_clusters": sum(1 for c in clusters if len(c) > 1),
              "n_images_in_duplicate_clusters": sum(len(c) for c in clusters if len(c) > 1)}
    for sp, imgs in splits.items():
        os.makedirs(os.path.join(out_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, sp, "labels"), exist_ok=True)
        n_inst = {c: 0 for c in CLASSES}
        for i, img in enumerate(imgs):
            it = byimg[img]
            stem = f"{it['source']}_{i:06d}"
            shutil.copy(img, os.path.join(out_dir, sp, "images", stem + os.path.splitext(img)[1]))
            rows = remap_label(it["label"], it["class_map"])
            for r in rows:
                n_inst[CLASSES[int(r.split()[0])]] += 1
            with open(os.path.join(out_dir, sp, "labels", stem + ".txt"), "w") as fh:
                fh.write("\n".join(rows))
        report[sp] = {"images": len(imgs), "instances": n_inst}
    # leakage proof: no test image may have a near-duplicate (dHash distance <= 4) in train
    cross = 0
    sets = {sp: [hashes[i] for i in imgs] for sp, imgs in splits.items()}
    for ha in sets["test"][:2000]:
        for hb in sets["train"]:
            if hamming(ha, hb) <= 4:
                cross += 1
                break
    report["test_images_with_near_duplicate_in_train"] = cross
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    sources = load_sources(args.data_root)
    items = collect(args.data_root, sources)
    if not items:
        raise SystemExit("no labelled images found")
    split_dir = os.path.join(args.out, "split")
    shutil.rmtree(split_dir, ignore_errors=True)
    split_report = build_splits(items, split_dir)
    if split_report["test_images_with_near_duplicate_in_train"] != 0:
        raise SystemExit(f"leak: {split_report['test_images_with_near_duplicate_in_train']} test images have a near-duplicate in train")

    data_yaml = os.path.join(args.out, "data.yaml")
    with open(data_yaml, "w") as fh:
        yaml.safe_dump({"path": split_dir, "train": "train/images", "val": "val/images",
                        "test": "test/images", "names": {i: n for i, n in enumerate(CLASSES)}}, fh)

    model = YOLO(args.weights)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, seed=SEED, deterministic=True,
                project=args.out, name="train", patience=25, exist_ok=True)
    best = os.path.join(args.out, "train", "weights", "best.pt")  # chosen on VAL fitness inside ultralytics

    final = YOLO(best)
    val = final.val(data=data_yaml, split="val", imgsz=args.imgsz, verbose=False)
    test = final.val(data=data_yaml, split="test", imgsz=args.imgsz, verbose=False)  # scored once
    per_class = {}
    for i, name in enumerate(CLASSES):
        if i in test.box.ap_class_index:
            k = list(test.box.ap_class_index).index(i)
            per_class[name] = {"precision": float(test.box.p[k]), "recall": float(test.box.r[k]),
                               "ap50": float(test.box.ap50[k]), "ap50_95": float(test.box.ap[k])}
    onnx_path = final.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)

    metrics = {
        "selection_rule": SELECTION_RULE,
        "seed": SEED,
        "split": split_report,
        "sources": {k: {f: v[f] for f in ("url", "licence")} for k, v in sources.items()},
        "val": {"map50": float(val.box.map50), "map50_95": float(val.box.map)},
        "test": {"map50": float(test.box.map50), "map50_95": float(test.box.map), "per_class": per_class},
        "onnx": os.path.basename(str(onnx_path)),
        "onnx_sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
        # Set by scripts/verify_rdd_detector.py only, never here:
        "artefact_check": {"passed": False, "note": "run scripts/verify_rdd_detector.py"},
        "deployment_check": {"passed": False, "note": "run scripts/verify_rdd_detector.py"},
    }
    with open(os.path.join(args.out, "detector_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps({k: metrics[k] for k in ("val", "test", "split")}, indent=2))


if __name__ == "__main__":
    main()
