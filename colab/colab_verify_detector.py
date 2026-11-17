# ROAD-SHIELD: artefact check for the newly trained detector (run in Colab AFTER training).
#
# Why this exists: the training script reports test mAP from Ultralytics running best.pt. The
# product, however, runs the exported best.onnx through OUR own decoder (letterbox, decode, NMS in
# models/road_damage_detector.py). The repository refuses to show boxes from a model until this
# has been checked (scripts/verify_rdd_detector.py, "artefact check"). best.pt and the held-out
# test photographs exist only inside this Colab runtime, so the check has to run here.
#
# Rule, copied unchanged from scripts/verify_rdd_detector.py and fixed before running:
#   served only if (a) the ONNX file's test mAP@0.5 is within 0.02 of best.pt's, and
#   (b) on the first 100 held-out test photographs our serving decoder recovers >= 90% of
#   best.pt's boxes and >= 90% of our boxes match a best.pt box (same class, IoU >= 0.5).
# If it fails, the failure is recorded as failed. Nothing is adjusted afterwards.
#
# HOW TO RUN: in the SAME Colab notebook (same runtime) as the training cell, after the training
# cell has printed DONE, paste this whole file into a NEW cell and run it. It needs about
# 5-10 minutes (the ONNX file is scored on the CPU). It then downloads artefact_check.json and
# best.pt to your computer.

import glob
import json
import os
import shutil
import sys
import tempfile

OUT = "/content/out"
REPO = "/content/repo"
RESULT = "/content/result"
IMGSZ = 640            # the size used in training
CONF = 0.25            # serving default (models/road_damage_detector.py)
IOU = 0.45
N_IMAGES = 100
CODES = ["D00", "D10", "D20", "D40"]   # serving codes for the 4 trained classes, in class order

BEST_PT = f"{OUT}/train/weights/best.pt"
BEST_ONNX = f"{OUT}/train/weights/best.onnx"
DATA_YAML = f"{OUT}/data.yaml"
METRICS = f"{OUT}/detector_metrics.json"
TEST_IMAGES = f"{OUT}/split/test/images"


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(ref, ours, thr=0.5):
    """Greedy one-to-one matching of (class, xyxy) lists; returns the matched count."""
    used, n = set(), 0
    for rc, rb in ref:
        best, bj = 0.0, None
        for j, (oc, ob) in enumerate(ours):
            if j in used or oc != rc:
                continue
            v = _iou(rb, ob)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= thr:
            used.add(bj)
            n += 1
    return n


def main(repo=REPO, out=OUT, result=RESULT, n_images=N_IMAGES):
    for p in (BEST_PT, BEST_ONNX, DATA_YAML, METRICS):
        if not os.path.exists(p.replace(OUT, out)):
            raise SystemExit(f"missing {p}. Run this in the same runtime, after the training cell finished.")
    sys.path.insert(0, repo)
    import cv2
    from ultralytics import YOLO
    from models.road_damage_detector import RoadDamageDetector

    best_pt, best_onnx = BEST_PT.replace(OUT, out), BEST_ONNX.replace(OUT, out)
    data_yaml, metrics_path = DATA_YAML.replace(OUT, out), METRICS.replace(OUT, out)
    test_images = TEST_IMAGES.replace(OUT, out)
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    pt_map50 = float((metrics.get("test") or {}).get("map50"))

    # The serving code loads damage_rdd2022_india.onnx + .json from a checkpoints folder.
    ck = tempfile.mkdtemp(prefix="ckpt_")
    shutil.copy(best_onnx, os.path.join(ck, "damage_rdd2022_india.onnx"))
    json.dump({"input_size": IMGSZ, "class_names": CODES, "conf_threshold": CONF, "iou_threshold": IOU},
              open(os.path.join(ck, "damage_rdd2022_india.json"), "w"))
    det = RoadDamageDetector(ck)
    det.verifying = True            # lets the detector run while its own checks are not recorded yet
    if det._session is None and getattr(det, "_cv_net", None) is None:
        raise SystemExit("the serving code could not load best.onnx")

    print("[artefact] scoring the exported ONNX file on the held-out test split (CPU) ...", flush=True)
    onnx_val = YOLO(best_onnx, task="detect").val(data=data_yaml, split="test", imgsz=IMGSZ, batch=1,
                                                   device="cpu", plots=False, verbose=False)
    onnx_map50 = round(float(onnx_val.box.map50), 4)

    ref_model = YOLO(best_pt)
    paths = sorted(glob.glob(os.path.join(test_images, "*.jpg")) + glob.glob(os.path.join(test_images, "*.png"))
                   + glob.glob(os.path.join(test_images, "*.jpeg")))[:n_images]
    if not paths:
        raise SystemExit(f"no test photographs found in {test_images}")
    n_ref = n_ours = m_ref = m_ours = 0
    for i, p in enumerate(paths):
        r = ref_model.predict(p, conf=CONF, iou=IOU, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        ref = [(CODES[int(c)], [float(v) for v in b]) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ours = []
        for d in det.detect(rgb):
            x, y, w, h = d["bbox_pixels"]
            ours.append((d["damage_code"], [x, y, x + w, y + h]))
        n_ref += len(ref)
        n_ours += len(ours)
        m_ref += _match(ref, ours)
        m_ours += _match(ours, ref)
        if (i + 1) % 25 == 0:
            print(f"  compared {i + 1}/{len(paths)} photographs", flush=True)

    recall = round(m_ref / n_ref, 4) if n_ref else 1.0
    precision = round(m_ours / n_ours, 4) if n_ours else 1.0
    passed = abs(onnx_map50 - pt_map50) <= 0.02 and recall >= 0.9 and precision >= 0.9
    res = {
        "onnx_test_map50": onnx_map50,
        "best_pt_test_map50": round(pt_map50, 4),
        "serving_decoder_vs_best_pt": {"photographs": len(paths), "best_pt_boxes": n_ref,
                                       "serving_boxes": n_ours, "recall": recall, "precision": precision},
        "rule": "ONNX test mAP@0.5 within 0.02 of best.pt, and the serving decoder recovers >= 90% of "
                "best.pt's boxes with >= 90% of its own matched (same class, IoU >= 0.5)",
        "passed": bool(passed),
        "onnx_sha256": metrics.get("onnx_sha256"),
    }
    os.makedirs(result, exist_ok=True)
    json.dump(res, open(os.path.join(result, "artefact_check.json"), "w", encoding="utf-8"), indent=2)
    shutil.copy(best_pt, os.path.join(result, "best.pt"))
    print(json.dumps(res, indent=2))
    print(f"\nARTEFACT CHECK {'PASSED' if passed else 'FAILED'}  (ONNX mAP50 {onnx_map50} vs best.pt {pt_map50:.4f}; "
          f"decoder recall {recall}, precision {precision} over {len(paths)} photographs)")
    return res


if __name__ == "__main__" or "google.colab" in sys.modules:
    res = main()
    try:
        from google.colab import files
        files.download(f"{RESULT}/artefact_check.json")
        files.download(f"{RESULT}/best.pt")
        print("Downloading artefact_check.json and best.pt (allow multiple downloads if asked).")
    except Exception as e:
        print("Automatic download did not start (", e, ") - download both from /content/result in the Files panel.")
