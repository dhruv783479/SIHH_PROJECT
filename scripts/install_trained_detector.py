"""
Install the detector trained with colab/colab_train_detector_v2.py as the candidate served detector.

    python -m scripts.install_trained_detector --metrics detector_metrics.json --onnx best.onnx \
           --artefact artefact_check.json
    python -m scripts.verify_rdd_detector --clean-roads        # then the second recorded check

What it does, and what it refuses to do
---------------------------------------
* The ONNX file must hash to the value the trainer recorded (`onnx_sha256`), so the file that is
  installed is the file whose test mAP was measured.
* Only measured values are copied. Missing fields stop the script; nothing is defaulted.
* The test split must have no near-duplicate in the training split (recorded by the trainer).
* The two checks that gate showing boxes (models/road_damage_detector.py) are written exactly as
  measured: the artefact check from Colab (passed or failed), the clean-road check as NOT YET RUN
  until `scripts.verify_rdd_detector --clean-roads` has run. Boxes stay hidden until both passed.
* The previous detector's records are kept inside the new report as "previous_model".
* The serving confidence is the loader's default (0.25). It was not tuned, and the report says so.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ENGINE_ROOT, "checkpoints")
ONNX_NAME, META_NAME, REPORT_NAME = ("damage_rdd2022_india.onnx", "damage_rdd2022_india.json",
                                     "road_damage_detector_report.json")
CODES = ["D00", "D10", "D20", "D40"]
RAW = {"0": "longitudinal_crack", "1": "transverse_crack", "2": "alligator_crack", "3": "pothole"}
SELECTION_RULE = ("best.pt = the epoch with the best Ultralytics validation fitness (0.1*mAP50 + 0.9*mAP50-95 "
                  "over all classes), early stop after 25 epochs without gain; the test split is scored once, "
                  "after the weights are fixed")
NOT_CLAIMED = ["zebra crossings (a separate pretrained detector is used as published)",
               "people and vehicles (a COCO-pretrained detector is used as published)",
               "accuracy on bus-mounted video; the test photographs are public-repository images",
               "box tightness beyond mAP@0.5:0.95 on the held-out test split"]


def _need(d, path):
    v = d
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v or v[k] is None:
            sys.exit(f"missing field '{path}' - refusing to install")
        v = v[k]
    return v


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
        fh.write("\n")


def main(argv=None, ckpt=CKPT):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="detector_metrics.json from the Colab run")
    ap.add_argument("--onnx", required=True, help="best.onnx from the same run")
    ap.add_argument("--artefact", help="artefact_check.json from colab/colab_verify_detector.py")
    a = ap.parse_args(argv)

    m = _load(a.metrics)
    for p in ("test.map50", "test.map50_95", "test.per_class", "val.map50", "val.map50_95", "onnx_sha256",
              "split.train.images", "split.val.images", "split.test.images", "sources"):
        _need(m, p)
    if _need(m, "split.test_images_with_near_duplicate_in_train") != 0:
        sys.exit("the test split has near-duplicates in the training split - not installing")
    sha = hashlib.sha256(open(a.onnx, "rb").read()).hexdigest()
    if sha != m["onnx_sha256"]:
        sys.exit(f"best.onnx does not match the file that was measured (sha256 {sha[:12]}... vs "
                 f"{str(m['onnx_sha256'])[:12]}...) - not installing")

    art = None
    if a.artefact:
        art = _load(a.artefact)
        if art.get("onnx_sha256") not in (None, m["onnx_sha256"]):
            sys.exit("artefact_check.json was made for a different ONNX file - not installing")
        for p in ("passed", "onnx_test_map50", "best_pt_test_map50", "serving_decoder_vs_best_pt"):
            _need(art, p)
    artefact_rec = art or {"passed": False, "note": "artefact check not run; run colab/colab_verify_detector.py"}
    deployment_rec = {"passed": False, "note": "not run yet: python -m scripts.verify_rdd_detector --clean-roads"}

    previous = {}
    for name in (REPORT_NAME, META_NAME):
        p = os.path.join(ckpt, name)
        if os.path.exists(p):
            previous[name] = {k: v for k, v in _load(p).items() if k != "previous_model"}

    test = m["test"]
    report = {
        "model": "YOLOv8s fine-tuned on RDD2022 India + Kaggle pothole (this repository's own run)",
        "weights_init": "yolov8s.pt (COCO-pretrained)",
        "input_size": 640, "classes": CODES, "raw_names": RAW,
        "data": {"photographs": {"train": m["split"]["train"]["images"], "val": m["split"]["val"]["images"],
                                 "test": m["split"]["test"]["images"]},
                 "test_instances": m["split"]["test"].get("instances"),
                 "near_duplicates_of_test_in_train": 0, "sources": m["sources"]},
        "selection_rule": SELECTION_RULE,
        "selection_rule_as_recorded_by_trainer": m.get("selection_rule"),
        "val": m["val"],
        "test": {"map50": test["map50"], "map50_95": test["map50_95"], "per_class": test["per_class"]},
        "serving_confidence": 0.25, "serving_confidence_chosen_on": "loader default; not tuned on any split",
        "onnx_sha256": m["onnx_sha256"], "seed": m.get("seed"),
        "not_claimed": NOT_CLAIMED,
        "artefact_check": artefact_rec, "deployment_check": deployment_rec,
        "previous_model": previous or None,
    }
    meta = {"model_name": report["model"], "input_size": 640, "class_names": CODES, "classes": CODES,
            "raw_names": RAW, "conf_threshold": 0.25, "iou_threshold": 0.45,
            "test": {"map50": test["map50"], "map50_95": test["map50_95"]},
            "onnx_sha256": m["onnx_sha256"],
            "artefact_check": artefact_rec, "deployment_check": deployment_rec}

    os.makedirs(ckpt, exist_ok=True)
    shutil.copy(a.onnx, os.path.join(ckpt, ONNX_NAME))
    _save(os.path.join(ckpt, META_NAME), meta)
    _save(os.path.join(ckpt, REPORT_NAME), report)
    print(f"installed: test mAP@0.5 {test['map50']:.4f}, mAP@0.5:0.95 {test['map50_95']:.4f}; "
          f"artefact check {'PASSED' if artefact_rec.get('passed') else 'NOT PASSED'}; clean-road check not run yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
