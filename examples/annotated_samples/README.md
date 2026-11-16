# Annotated sample images (5)

Boxes for potholes (incl. water-filled, 30 of them across 4 images; 27 on the many-potholes road, the farthest only 3-5 px tall so accurate to a few pixels), cars, a person and a zebra crossing.

| Folder | Contents |
|---|---|
| `images/` | the 5 source photos (PNG) |
| `labels/` | YOLO-format labels (`cls cx cy w h`, normalised); classes in `annotations.json` |
| `overlays/` | the photos with the reference boxes drawn |
| `annotations.json` | boxes in pixels with the origin of each (`hand` or `coco`) |
| `evaluate_samples.py` | scores the shipped detectors against these boxes |

## Honest status
* **Not a training set.** Five images cannot train or fairly test a detector; training on them and testing on
  them would be circular. Use them as a smoke test and as five extra labelled examples for the real run
  (`training/train_detectors_real.py`), kept out of the test split.
* **How the boxes were made.** Pothole and zebra boxes were placed by hand from a pixel grid (about +/-10 px).
  Car and person boxes were seeded from the COCO YOLO model (conf >= 0.5) and checked by eye, so their scores
  below are optimistic. A human should verify all boxes before any number is reported.
* **Current detector results on these images (IoU >= 0.5):**

| Class | Reference boxes | Found | False boxes |
|---|---|---|---|
| pothole | 30 | 0 | 0 |
| car | 10 | 10 | 0 |
| person | 1 | 1 | 0 |
| zebra_crossing | 1 | 0 | 1 |

  The shipped pothole and crosswalk weights do not detect these cases. Fixing that needs the retraining on real
  labelled data described in `training/`, not more post-processing.
* **Licence.** Several images carry stock-photo watermarks. Confirm you may redistribute them before publishing
  this folder in a public repository; otherwise keep only `labels/` and `annotations.json`.
