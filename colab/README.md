# Colab runbook: train and verify the road-damage detector

Both scripts are written to be pasted whole into a Google Colab code cell (T4 GPU runtime).

1. `colab_train_detector_v2.py` downloads RDD2022 India (CC BY 4.0), reads the Kaggle pothole
   zip you uploaded to `/content/archive.zip` (DbCL v1.0), prepares the data, trains YOLOv8s for
   up to 100 epochs on a duplicate-safe split, scores the test split once, and downloads
   `detector_metrics.json` and `best.onnx`. Expect about 3 hours on a T4.
2. `colab_verify_detector.py` runs in the SAME runtime afterwards. It checks that the exported
   ONNX file, run through this repository's own decoder (`models/road_damage_detector.py`),
   behaves like `best.pt` (rule printed at the top of the script and fixed beforehand), and
   downloads `artefact_check.json` and `best.pt`. `best.pt` and the test photographs exist only
   inside the runtime, so this check cannot be done later.

The detector stays blocked in the product until both recorded checks pass
(`scripts/verify_rdd_detector.py`: artefact check, then clean-road check). Only measured numbers
go into the report; a failed check is recorded as failed.

Note on the metrics file: `selection_rule` in a `detector_metrics.json` written before commit
"Selection rule wording" describes the rule as pothole mAP50, which is not what Ultralytics does.
The weights are chosen on Ultralytics' validation fitness (0.1*mAP50 + 0.9*mAP50-95, all classes).
