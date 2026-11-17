# ROAD-SHIELD: where does the 0.0211 mAP gap between best.pt and best.onnx come from?
# (run in Colab, in the SAME runtime that trained the detector; needs about 15-25 minutes on CPU)
#
# Background. The recorded artefact check FAILED: the exported ONNX scored test mAP@0.5 0.4216, the
# trained network (best.pt) 0.4005, a gap of 0.0211 against the limit of 0.02. Scoring best.pt on the CPU
# with batch 1 gave 0.3996, so device and batch size do not explain it.
#
# Hypothesis tested here (written before running): Ultralytics scores a .pt model with "rect" batches
# (each photograph padded as little as possible) but scores a static ONNX file with square padding to
# 640 x 640, because it forces rect=False for export formats. The padding changes what the network sees.
# If that is the cause, best.pt scored WITH square padding should land close to the ONNX figure.
#
# What this script does:  it only measures. It scores, on the same held-out test photographs,
#   A  best.pt, default (rect) scoring          - the protocol that produced the recorded 0.4005
#   B  best.pt, square 640 x 640 padding        - the padding the ONNX file gets
#   C  best.onnx, as exported                   - what the product runs
#   D  best.pt re-exported with dynamic shapes, scored with rect (control; optional)
# Pre-fixed reading rule: "padding explains the gap" only if |C - B| <= 0.005 AND |A - B| >= 0.015.
# Otherwise the result is "not explained" and nothing is concluded.
#
# What it does NOT do:  it does not rewrite the artefact check, does not edit any repository file and
# never reports "passed". The pre-registered rule (ONNX within 0.02 of best.pt's recorded mAP, decoder
# recall and precision >= 90%) stays exactly as it was, so the recorded result stays FAILED. Whether the
# team wants to amend the rule for the NEXT model is a decision for people, made in writing and dated.
#
# HOW TO RUN: paste this whole file into a NEW cell of the training notebook (same runtime) and run it.
# It downloads export_gap_diagnosis.json; send that file back.

import json
import os
import sys

OUT = "/content/out"
RESULT = "/content/result"
IMGSZ = 640
THRESH_CLOSE = 0.005             # |C - B| must be at most this
THRESH_EFFECT = 0.015            # |A - B| must be at least this
RULE_LIMIT = 0.02                # the pre-registered limit, copied unchanged

BEST_PT = f"{OUT}/train/weights/best.pt"
BEST_ONNX = f"{OUT}/train/weights/best.onnx"
DATA_YAML = f"{OUT}/data.yaml"
METRICS = f"{OUT}/detector_metrics.json"


def reading(a, b, c, d=None):
    """Turn four mAP@0.5 values into a verdict. Pure function (no I/O) so it can be tested."""
    out = {
        "A_best_pt_rect": a, "B_best_pt_square": b, "C_onnx": c, "D_dynamic_onnx_rect": d,
        "rect_effect_A_minus_B": round(a - b, 4),
        "onnx_vs_square_best_pt_C_minus_B": round(c - b, 4),
    }
    explained = abs(c - b) <= THRESH_CLOSE and abs(a - b) >= THRESH_EFFECT
    out["padding_explains_gap"] = bool(explained)
    out["reading"] = (
        "The gap is a scoring-protocol difference: best.pt scored with square 640 x 640 padding (the padding "
        "the ONNX file gets) lands within 0.005 of the ONNX figure, while the default rect scoring differs by "
        "0.015 or more. The exported file does not behave differently from the network."
        if explained else
        "Not explained by padding under the pre-fixed reading rule. No conclusion is drawn about the cause.")
    return out


def main(out=OUT, result=RESULT):
    for p in (BEST_PT, BEST_ONNX, DATA_YAML, METRICS):
        p = p.replace(OUT, out)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}. Run this in the SAME Colab runtime as the training cell. If the runtime "
                             "was reset, the held-out test photographs are gone: say so, and do not rebuild them "
                             "by guessing.")
    from ultralytics import YOLO
    metrics = json.load(open(METRICS.replace(OUT, out), encoding="utf-8"))
    recorded = float((metrics.get("test") or {}).get("map50"))
    data = DATA_YAML.replace(OUT, out)
    pt = BEST_PT.replace(OUT, out)
    onnx = BEST_ONNX.replace(OUT, out)

    def score(model_path, rect, label, task=None):
        print(f"[{label}] scoring ...", flush=True)
        m = YOLO(model_path, task=task) if task else YOLO(model_path)
        v = m.val(data=data, split="test", imgsz=IMGSZ, batch=1, device="cpu", rect=rect, plots=False, verbose=False)
        r = round(float(v.box.map50), 4)
        print(f"[{label}] mAP@0.5 = {r}", flush=True)
        return r

    a = score(pt, True, "A best.pt rect")
    b = score(pt, False, "B best.pt square")
    c = score(onnx, False, "C best.onnx", task="detect")
    d = None
    try:
        dyn = YOLO(pt).export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=False, opset=12)
        d = score(dyn, True, "D dynamic ONNX rect", task="detect")
    except Exception as e:                      # the control is optional
        print("control D skipped:", e)

    res = reading(a, b, c, d)
    res.update({
        "recorded_best_pt_test_map50": round(recorded, 4),
        "rule_limit": RULE_LIMIT,
        "rule_v1_as_recorded": "ONNX test mAP@0.5 within 0.02 of best.pt's recorded mAP, decoder recall and precision >= 90%",
        "gap_under_recorded_rule": round(abs(c - recorded), 4),
        "recorded_result": "FAILED" if abs(c - recorded) > RULE_LIMIT else "check the other conditions",
        "like_for_like_gap_C_vs_B": round(abs(c - b), 4),
        "note": ("This file does not change the recorded artefact check and is not a pass. A rule amended after a "
                 "result is seen is a post-hoc change; if the team wants a like-for-like rule it must be written "
                 "down and dated as an amendment, and applied openly."),
    })
    os.makedirs(result, exist_ok=True)
    path = os.path.join(result, "export_gap_diagnosis.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=2)
    print(json.dumps(res, indent=2))
    return path


if __name__ == "__main__" or "google.colab" in sys.modules:
    p = main()
    try:
        from google.colab import files
        files.download(p)
    except Exception as e:
        print("Automatic download did not start (", e, ") - download", p, "from the Files panel.")
