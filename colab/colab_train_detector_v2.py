# ROAD-SHIELD: real-data detector training (v2), one script for a GOOGLE COLAB notebook.
#
# Before you run it:
#   1. Runtime menu -> Change runtime type -> Hardware accelerator = T4 GPU -> Save.
#   2. Upload your Kaggle download "archive.zip" (the andrewmvd/pothole-detection dataset)
#      to /content using the Files panel on the left (folder icon -> upload icon). Wait until
#      the upload bar finishes. If you do not have the zip, see KAGGLE CREDENTIALS below.
#   3. Paste this whole file into ONE code cell and run it. Keep the browser tab open.
#
# Data (all human-labelled boxes):
#   * RDD2022 India  - the official CRDDC-2022 release (Indian roads, smartphone photos)
#   * andrewmvd/pothole-detection (Kaggle, 665 images)  - extra variety
# Pothole is the class that matters; the three crack classes come along from RDD2022.
#
# Honest-evaluation design (unchanged): near-duplicate photos are clustered before the
# split so a copy never sits on both sides; the model-selection rule is fixed before the
# test set is scored; the test set is scored once; only measured numbers are written.
# At the end the two result files download to your computer by themselves; they are also
# kept in /content/result (visible in the Files panel).
#
# KAGGLE CREDENTIALS (only if you do NOT upload archive.zip): in Colab click the key icon
# (Secrets), add two secrets named KAGGLE_USERNAME and KAGGLE_KEY, switch "Notebook access"
# on for both, then run. Create the key at kaggle.com -> Settings -> API -> Create New Token.

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
DATA = f"{WORK}/data"
OUT = f"{WORK}/out"
RESULT = f"{WORK}/result"
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
   f"--epochs {EPOCHS} --weights {WEIGHTS} --imgsz {IMGSZ}")

# ---- 6. results ---------------------------------------------------------------------
os.makedirs(RESULT, exist_ok=True)
metrics = f"{OUT}/detector_metrics.json"
onnx = f"{OUT}/train/weights/best.onnx"
for p in (metrics, onnx):
    if not os.path.exists(p):
        raise SystemExit(f"expected output not found: {p}")
shutil.copy(metrics, RESULT)
shutil.copy(onnx, RESULT)
print("\n=== detector_metrics.json ===")
print(json.dumps(json.load(open(metrics, encoding="utf-8")), indent=2)[:6000])
print(f"\nDONE. Files are in {RESULT}: detector_metrics.json and best.onnx")

# ---- 6b. keep what later checks need (a Colab reset deletes /content) -----------------
# best.pt, data.yaml and the held-out test photographs are needed by colab_diagnose_export_gap.py and by
# the v3 experiment. An earlier run kept only the ONNX file, so those checks could not be run afterwards.
# Recorded v2 split, copied from checkpoints/road_damage_detector_report.json: a retrained run is only
# "the same split" if its counts agree. Equal counts are strong evidence, not proof photo by photo.
RECORDED_SPLIT = {"train": 6248, "val": 781, "test": 780,
                  "test_instances": {"longitudinal_crack": 155, "transverse_crack": 8,
                                     "alligator_crack": 198, "pothole": 523}}
_sp = json.load(open(metrics, encoding="utf-8")).get("split", {})
_got = {"train": _sp.get("train", {}).get("images"), "val": _sp.get("val", {}).get("images"),
        "test": _sp.get("test", {}).get("images"),
        "test_instances": _sp.get("test", {}).get("instances")}
print("\nSplit counts this run:", _got)
print("Recorded v2 split:    ", RECORDED_SPLIT)
print("SPLIT COUNTS MATCH THE RECORDED v2 SPLIT" if _got == RECORDED_SPLIT else
      "WARNING: split counts DIFFER from the recorded v2 split - this is a different test set; do not compare "
      "its numbers with the recorded 0.4005 as if they were the same measurement.")
try:
    from google.colab import drive
    drive.mount("/content/drive")
    keep = "/content/drive/MyDrive/road_shield_v2_artifacts"
    os.makedirs(keep, exist_ok=True)
    for src in (f"{OUT}/train/weights/best.pt", onnx, metrics, f"{OUT}/data.yaml"):
        shutil.copy(src, keep)
    shutil.rmtree(f"{keep}/split_test", ignore_errors=True)
    shutil.copytree(f"{OUT}/split/test", f"{keep}/split_test")
    print("Saved best.pt, best.onnx, detector_metrics.json, data.yaml and the test photographs to Google Drive:", keep)
except Exception as e:
    print("Could not save to Google Drive (", e, ") - download /content/out/train/weights/best.pt yourself now.")

try:
    from google.colab import files
    files.download(f"{RESULT}/detector_metrics.json")
    files.download(f"{RESULT}/best.onnx")
    print("Your browser should now be downloading both files (allow multiple downloads if asked).")
except Exception as e:
    print("Automatic download did not start (", e, ") - right-click the files in /content/result "
          "in the Files panel and choose Download.")
