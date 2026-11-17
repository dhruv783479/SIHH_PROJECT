# ROAD-SHIELD: detector experiment v3 - more transverse-crack (D10) training data, judged on a FROZEN test set.
# One script for a GOOGLE COLAB notebook. It does NOT replace anything by itself: it produces numbers and a file.
#
# Why: the v2 detector's held-out India test set holds only 8 transverse-crack boxes, so its D10 score says
# nothing. More D10 examples come from the labelled train folders of other RDD2022 countries. They are a
# different domain (other roads, cameras, markings), so they are used for TRAINING and are reported per source;
# they are never presented as India performance.
#
# The comparison is only fair if the model is judged on photographs it never saw. So the v2 test photographs are
# FROZEN: this script removes every copy and near-copy of them (dHash <= 4 bits, the same rule as the split)
# from the training pool, trains the new model, and scores it on exactly those 780 photographs. The v2 weights
# were also never trained on them, so both models face the same unseen photographs. Run it in the SAME Colab
# runtime that trained v2 (the frozen photographs are in /content/out/split/test/images); it refuses to run if
# they are gone, because rebuilding "the same" test set by guessing would defeat the point.
#
# Rule for adopting v3 instead of v2 (written before anything is run, never edited afterwards):
#   v3 is a candidate for serving only if ALL hold:
#     1. on the frozen v2 test photographs its mAP@0.5:0.95 is >= v2's recorded 0.1863 minus 0.01
#        (no meaningful regression on what the product is for), AND
#     2. its pothole AP@0.5 on those photographs is >= v2's recorded pothole AP@0.5 minus 0.02, AND
#     3. it then passes the artefact check (colab/colab_verify_detector.py) and the clean-road check
#        (scripts/verify_rdd_detector.py --clean-roads) exactly as v2 had to.
#   A better transverse-crack AP is REPORTED, with the number of boxes behind it, but cannot by itself justify
#   adoption: with so few India D10 boxes it cannot be certified for India.
#
# HOW TO RUN: (1) merge the pull request that added this script, so the repository's master has it.
# (2) Runtime -> T4 GPU. Same runtime as the v2 training, same upload of archive.zip as v2. (3) Paste this whole
# file into ONE new cell and run it. About 2-3 hours (13 GB RDD2022 download, then training).
# Output: result_v3/detector_v3_report.json (download is automatic) and best.onnx.

import json
import os
import shutil
import subprocess
import sys
import zipfile

# ----------------------------------------------------------------------------------
# 1. EDIT THESE
# ----------------------------------------------------------------------------------
USE_RDD2022_INDIA = True            # large one-off download from figshare (10-25 min)
EXTRA_COUNTRIES = ["Japan", "Czech", "Norway", "United_States"]   # labelled train folders, training only
FROZEN_TEST = "/content/out/split/test/images"   # v2 test photographs (same runtime)
V2_RECORDED = {"map50_95": 0.1863287, "pothole_ap50": None}   # pothole AP read from /content/out/detector_metrics.json
POTHOLE_ZIP = "/content/archive.zip"            # your uploaded Kaggle download
POTHOLE_SLUG = "andrewmvd/pothole-detection"    # used only when the zip is not uploaded
POTHOLE_URL = "https://www.kaggle.com/datasets/andrewmvd/pothole-detection"
# (the second dataset I suggested earlier, chitholian/annotated-potholes-dataset, is a
#  re-upload of the first one: 663 of 665 images match, so it is left out on purpose)

# Real licence of every source, copied from its page. Keys are the source names printed
# when the script stops: "rdd2022_india", "andrewmvd__pothole-detection", ...
# Training refuses to run while a licence is a placeholder.
LICENCES = {
    "rdd2022_india": "CC BY 4.0 (figshare record 10.6084/m9.figshare.21431547.v1, licence field read from the figshare API on 17-11-2026)",
    "andrewmvd__pothole-detection": "Database Contents License (DbCL) v1.0, as stated in the description on the Kaggle page (sidebar field reads 'Other (specified in description)'), read 17-11-2026; credit the original authors: Make ML, Potholes Dataset, https://makeml.app/datasets/potholes",
}

for _c in EXTRA_COUNTRIES:
    LICENCES[f"rdd2022_{_c.lower()}"] = LICENCES["rdd2022_india"]   # same figshare record, same licence

MAPS = []                           # e.g. ["hole=pothole"] if a class name is not recognised
IGNORE_UNMAPPED = True              # classes that are not pothole/crack types are dropped
                                    # (printed below); photos holding ONLY such classes are skipped
INCLUDE_ANNOTATED_SAMPLES = False   # your 5 hand-annotated photos; needs LICENCES["annotated_samples"]

EPOCHS = 100
WEIGHTS = "yolov8s.pt"              # "yolov8m.pt" is stronger but roughly twice as slow
IMGSZ = 640                         # 800 or 960 helps small potholes, at more time and memory
BRANCH = "master"
# ----------------------------------------------------------------------------------

WORK = "/content"
REPO = f"{WORK}/repo"
DATA = f"{WORK}/data_v3"
OUT = f"{WORK}/out_v3"
RESULT = f"{WORK}/result_v3"
RDD_DIR = f"{WORK}/rdd2022"
POTHOLE_RAW = f"{WORK}/pothole_raw"
FIGSHARE_PAGE = "https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547"


def sh(cmd):
    """Run a shell command and stream its output into the notebook (a plain
    subprocess.run hides it, which is how an earlier failure showed no reason)."""
    print("\n$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        tail = tail[-40:]
    if p.wait() != 0:
        raise SystemExit(f"FAILED (exit {p.returncode}): {cmd}\n--- last lines ---\n" + "".join(tail))


if not os.path.isdir(FROZEN_TEST) or not os.listdir(FROZEN_TEST):
    raise SystemExit(f"{FROZEN_TEST} is missing. This script must run in the runtime that trained v2: the frozen "
                     "test photographs live there. Do not rebuild them by guessing; say that the runtime was lost.")
_v2m = "/content/out/detector_metrics.json"
if os.path.exists(_v2m):
    _m = json.load(open(_v2m, encoding="utf-8"))
    V2_RECORDED["map50_95"] = float(_m["test"]["map50_95"])
    V2_RECORDED["pothole_ap50"] = float(_m["test"]["per_class"]["pothole"]["ap50"])
if V2_RECORDED["pothole_ap50"] is None:
    raise SystemExit("could not read v2's recorded numbers from /content/out/detector_metrics.json")

# ---- 2. setup ---------------------------------------------------------------------
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)   # a re-run may start inside the repo folder that is about to be deleted
sh("pip -q install ultralytics opencv-python-headless pyyaml kagglehub")
REUSE_DATA = os.path.exists(f"{DATA}/sources.json")   # skip re-preparing data on a re-run
sh(f"rm -rf {REPO} {OUT} " + ("" if REUSE_DATA else f"{DATA} ") + f"&& git clone --depth 1 -b {BRANCH} "
   f"https://github.com/udbhav968-creator/SIH_PROJECT.git {REPO}")
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch
print("GPU available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime menu -> Change runtime type -> T4 GPU -> Save, then run again.")

# ---- 3. real labelled data --------------------------------------------------------
prep = f"python training/prepare_kaggle_detection_data.py --out {DATA}"

if os.path.exists(POTHOLE_ZIP):
    # Uses the download you already have: no Kaggle login needed. The source name is the
    # same one the Kaggle route would give, so the licence entry above applies to it.
    if not os.path.isdir(POTHOLE_RAW):
        print("Unzipping", POTHOLE_ZIP)
        with zipfile.ZipFile(POTHOLE_ZIP) as z:
            z.extractall(POTHOLE_RAW)
    prep += f' --local "andrewmvd__pothole-detection,{POTHOLE_RAW},{POTHOLE_URL}"'
else:
    try:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    except Exception as e:
        raise SystemExit(
            f"{POTHOLE_ZIP} was not found and the Kaggle secrets could not be read ({e}).\n"
            "Either upload your archive.zip to /content (Files panel), or add the secrets "
            "KAGGLE_USERNAME and KAGGLE_KEY (key icon on the left, Notebook access ON).")
    prep += f" --dataset {POTHOLE_SLUG}:pothole"

if USE_RDD2022_INDIA:
    import scripts.fetch_rdd2022_india as rdd
    rdd.OUT = RDD_DIR
    os.makedirs(RDD_DIR, exist_ok=True)
    india = rdd.india_root() or rdd.main([])
    train_dir = os.path.join(india, "train")
    print("RDD2022 India train folder:", train_dir)
    prep += f' --local "rdd2022_india,{train_dir},{FIGSHARE_PAGE}"'

if EXTRA_COUNTRIES:
    import scripts.fetch_rdd2022_countries as fc
    fc.OUT = RDD_DIR
    for c, root in fc.main(EXTRA_COUNTRIES).items():
        prep += f' --local "rdd2022_{c.lower()},{os.path.join(root, "train")},{FIGSHARE_PAGE}"'

for m in MAPS:
    prep += f' --map "{m}"'
if IGNORE_UNMAPPED:
    prep += " --ignore-unmapped"
if REUSE_DATA:
    print("Reusing already-prepared data in", DATA, "(delete it to rebuild)")
else:
    sh(prep)
    if INCLUDE_ANNOTATED_SAMPLES:
        sh(f"python training/export_samples_as_source.py --data-root {DATA}")

# ---- 4. licences (training refuses placeholders) ----------------------------------
sp = f"{DATA}/sources.json"
sources = json.load(open(sp, encoding="utf-8"))
for name, lic in LICENCES.items():
    if name in sources:
        sources[name]["licence"] = lic
    else:
        print(f"note: LICENCES has '{name}' but no such source was prepared")
json.dump(sources, open(sp, "w", encoding="utf-8"), indent=2)

missing = [k for k, v in sources.items()
           if "SEE KAGGLE PAGE" in str(v.get("licence", "")) or "CHECK " in str(v.get("licence", ""))]
print("\nSources prepared:")
for k, v in sources.items():
    print(f"  {k}: {v.get('n_images')} images | {v['url']} | licence: {v['licence']}")
if missing:
    raise SystemExit(
        "\nSTOP: fill in the real licence for these sources in LICENCES (above), then run again:\n  "
        + "\n  ".join(f"{k}  ->  {sources[k]['url']}" for k in missing))

# ---- 5. train + test (grouped split, rule fixed before test scoring) ---------------
sh(f"python training/train_detectors_real.py --data-root {DATA} --out {OUT} "
   f"--epochs {EPOCHS} --weights {WEIGHTS} --imgsz {IMGSZ} --exclude-near-duplicates-of {FROZEN_TEST}")

# ---- 6. judge on the frozen photographs, per source, and write the report ---------------
import yaml
from ultralytics import YOLO
from training import score_by_source as sbs

best = f"{OUT}/train/weights/best.pt"
model = YOLO(best)
CLASSES = sbs.CLASSES


def per_class_of(v):
    idx = [int(i) for i in v.box.ap_class_index]
    return {CLASSES[i]: {"ap50": round(float(v.box.ap50[k]), 4), "ap50_95": round(float(v.box.ap[k]), 4)}
            for k, i in enumerate(idx)}


# frozen v2 test photographs: the folder's labels sit next to it (split/test/labels)
frozen_yaml = f"{OUT}/frozen_test.yaml"
frozen_lbl_root = os.path.dirname(os.path.dirname(FROZEN_TEST))        # /content/out/split
yaml.safe_dump({"path": frozen_lbl_root, "train": "test/images", "val": "test/images", "test": "test/images",
                "names": {i: n for i, n in enumerate(CLASSES)}}, open(frozen_yaml, "w"))
fz = model.val(data=frozen_yaml, split="test", imgsz=IMGSZ, verbose=False, plots=False)
frozen = {"photographs": len(os.listdir(FROZEN_TEST)),
          "map50": round(float(fz.box.map50), 4), "map50_95": round(float(fz.box.map), 4),
          "per_class": per_class_of(fz)}
# the frozen set holds India and Kaggle photographs; count the boxes per source and class
frozen_counts = sbs.count_boxes(frozen_lbl_root)

# v3's own pooled test split, cut by source (NOT comparable with v2: different photographs)
own = {}
for src, n in sbs.count_boxes(f"{OUT}/split").items():
    ymlp, k = sbs.write_source_yaml(f"{OUT}/split", src, f"{OUT}/lists")
    v = model.val(data=ymlp, split="test", imgsz=IMGSZ, verbose=False, plots=False)
    own[src] = {"photographs": k, "boxes": n["boxes"], "map50": round(float(v.box.map50), 4),
                "map50_95": round(float(v.box.map), 4), "per_class": per_class_of(v)}
    low = [c for c, b in n["boxes"].items() if 0 < b < 30]
    own[src]["warning"] = ("fewer than 30 test boxes for: " + ", ".join(low) + " - those AP values are not meaningful") if low else None

c1 = frozen["map50_95"] >= V2_RECORDED["map50_95"] - 0.01
pot = (frozen["per_class"].get("pothole") or {}).get("ap50")
c2 = pot is not None and pot >= V2_RECORDED["pothole_ap50"] - 0.02
report = {
    "what": "v3: v2's data plus labelled train folders of " + ", ".join(EXTRA_COUNTRIES) + "; judged on the frozen v2 test photographs",
    "extra_countries_are_training_only": True,
    "frozen_test_v2_photographs": frozen,
    "frozen_test_boxes_per_source": frozen_counts,
    "v2_recorded": V2_RECORDED,
    "own_pooled_test_by_source": own,
    "adoption_rule_conditions": {
        "1_map50_95_not_worse_than_v2_minus_0.01": bool(c1),
        "2_pothole_ap50_not_worse_than_v2_minus_0.02": bool(c2),
        "3_artefact_and_clean_road_checks": "not run here: they run on this model's best.onnx afterwards",
    },
    "candidate_for_serving": bool(c1 and c2),
    "note": "candidate_for_serving only means conditions 1 and 2 hold; v3 is NOT served until condition 3 passes. "
            "Transverse-crack AP on the frozen photographs rests on very few boxes (see frozen_test_boxes_per_source).",
}
os.makedirs(RESULT, exist_ok=True)
rp = f"{RESULT}/detector_v3_report.json"
json.dump(report, open(rp, "w", encoding="utf-8"), indent=2)
shutil.copy(f"{OUT}/detector_metrics.json", f"{RESULT}/detector_metrics_v3.json")
shutil.copy(f"{OUT}/train/weights/best.onnx", f"{RESULT}/best_v3.onnx")
shutil.copy(best, f"{RESULT}/best_v3.pt")
print(json.dumps(report, indent=2)[:9000])
print(f"\nDONE. Files are in {RESULT}")

try:
    from google.colab import files
    for f in ("detector_v3_report.json", "detector_metrics_v3.json", "best_v3.onnx", "best_v3.pt"):
        files.download(f"{RESULT}/{f}")
    print("Your browser should now be downloading four files (allow multiple downloads if asked).")
except Exception as e:
    print("Automatic download did not start (", e, ") - right-click the files in /content/result_v3 and choose Download.")
