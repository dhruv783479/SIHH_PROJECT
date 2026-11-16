"""Score the shipped detectors on the 5 annotated sample images (smoke test, NOT a benchmark).
Run from the repo root:  PYTHONPATH=. python examples/annotated_samples/evaluate_samples.py
Reports per-class recall and precision at IoU>=0.5 against annotations.json."""
import json, os, sys
import cv2
from models.onnx_object_detector import ONNXObjectDetector
from models.road_damage_detector import RoadDamageDetector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0.0


def xyxy(b): return [b[0], b[1], b[0]+b[2], b[1]+b[3]]


def main():
    ann = json.load(open(os.path.join(HERE, "annotations.json")))
    ck = os.path.join(ROOT, "checkpoints")
    coco = ONNXObjectDetector(checkpoints_dir=ck, conf_threshold=0.5)
    cw = ONNXObjectDetector(checkpoints_dir=ck, weights_path=os.path.join(ck, "crosswalk_rdd.onnx"),
                            class_names=["Zebra Crosswalk", "Road Guide Arrow"], conf_threshold=0.25)
    dmg = RoadDamageDetector(checkpoints_dir=ck)
    tp, fn, fp = {}, {}, {}
    for name, d in ann["images"].items():
        rgb = cv2.cvtColor(cv2.imread(os.path.join(HERE, "images", name + ".png")), cv2.COLOR_BGR2RGB)
        pred = {"pothole": [xyxy(x["bbox_pixels"]) for x in dmg.detect(rgb) if x.get("damage_code") == "D40"],
                "car": [xyxy(x["bbox_pixels"]) for x in coco.detect(rgb) if x["class_name"] == "car"],
                "person": [xyxy(x["bbox_pixels"]) for x in coco.detect(rgb) if x["class_name"] == "person"],
                "zebra_crossing": [xyxy(x["bbox_pixels"]) for x in cw.detect(rgb) if x["class_name"] == "Zebra Crosswalk"]}
        for c in ann["classes"]:
            gts = [b[1:5] for b in d["boxes"] if b[0] == c]; used = set()
            for p in pred[c]:
                best = max(((iou(p, g), j) for j, g in enumerate(gts) if j not in used), default=(0, -1))
                if best[0] >= 0.5: used.add(best[1]); tp[c] = tp.get(c, 0) + 1
                else: fp[c] = fp.get(c, 0) + 1
            fn[c] = fn.get(c, 0) + len(gts) - len(used)
    print(f"{'class':16s} TP  FN  FP  recall precision")
    for c in ann["classes"]:
        t, f, p = tp.get(c, 0), fn.get(c, 0), fp.get(c, 0)
        print(f"{c:16s} {t:3d} {f:3d} {p:3d}  {t/(t+f) if t+f else 0:6.2f} {t/(t+p) if t+p else 0:9.2f}")
    print("NOTE: car/person reference boxes were seeded from the COCO model itself, so those two rows are optimistic; "
          "pothole and zebra references are hand-placed and independent of the models.")


if __name__ == "__main__":
    sys.exit(main())
