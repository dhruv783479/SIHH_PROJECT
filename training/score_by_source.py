"""
Per-source scoring helpers for the detector's held-out test split.

The split written by training/train_detectors_real.py names each file "<source>_<index>.jpg", so a pooled test
split can be cut back into one test list per data source (for example India alone) without touching the
labels. That is what lets a model trained on several countries still be judged on India photographs only, and
what shows how many test boxes of each class each source really contributes.

No Ultralytics import here: the Colab script feeds the lists to model.val(). Everything below is plain file
and text handling, so it is unit-tested.
"""
import glob
import os

CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def source_of(stem):
    """'rdd2022_india_000123' -> 'rdd2022_india' (the index after the last underscore is dropped)."""
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


def group_by_source(split_dir, split="test"):
    """{source: [image paths]} for one split of the grouped split folder."""
    groups = {}
    for p in sorted(glob.glob(os.path.join(split_dir, split, "images", "*"))):
        if p.lower().endswith(IMG_EXT):
            groups.setdefault(source_of(os.path.splitext(os.path.basename(p))[0]), []).append(p)
    return groups


def count_boxes(split_dir, split="test"):
    """{source: {class name: number of labelled boxes}} read from the YOLO label files."""
    out = {}
    for src, imgs in group_by_source(split_dir, split).items():
        n = {c: 0 for c in CLASSES}
        for p in imgs:
            lbl = os.path.join(split_dir, split, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
            if not os.path.exists(lbl):
                continue
            for line in open(lbl, encoding="utf-8").read().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(float(parts[0])) < len(CLASSES):
                    n[CLASSES[int(float(parts[0]))]] += 1
        out[src] = {"photographs": len(imgs), "boxes": n}
    return out


def write_source_yaml(split_dir, source, out_dir, split="test"):
    """Write <out_dir>/<source>_<split>.txt (image list) and a data yaml that points 'test' at it.
    Ultralytics accepts a .txt list of image paths; labels are found next to the images' labels folder."""
    import yaml
    imgs = group_by_source(split_dir, split).get(source, [])
    if not imgs:
        raise ValueError(f"no {split} photographs for source '{source}'")
    os.makedirs(out_dir, exist_ok=True)
    lst = os.path.join(out_dir, f"{source}_{split}.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(imgs) + "\n")
    ymlp = os.path.join(out_dir, f"{source}_{split}.yaml")
    with open(ymlp, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": split_dir, "train": lst, "val": lst, "test": lst,
                        "names": {i: n for i, n in enumerate(CLASSES)}}, fh)
    return ymlp, len(imgs)
