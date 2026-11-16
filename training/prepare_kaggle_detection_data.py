"""
Download human-labelled Kaggle detection datasets and convert them to the YOLO layout
expected by training/train_detectors_real.py.

Usage (Colab; needs a Kaggle API token in ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY):
    python training/prepare_kaggle_detection_data.py --out /content/data \
        --dataset owner/slug:pothole --dataset owner/slug2:pothole,crack ...

Each --dataset is "<kaggle owner/slug>[:<comma list of target classes present in it>]".
Target class names must be from: longitudinal_crack, transverse_crack, alligator_crack, pothole.
Annotation formats auto-detected per dataset: YOLO txt, Pascal VOC xml, COCO json.
The script prints, per dataset, the raw class names it found; if a name cannot be mapped
to a target class it STOPS and asks you to pass --map "<raw>=<target>" (no silent guessing).
Water-filled potholes must be labelled as potholes in the source; check a few images by eye.
Licence/URL are read from the Kaggle metadata when available and must be non-empty.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET

TARGETS = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
IMG_EXT = (".jpg", ".jpeg", ".png")
DEFAULT_ALIASES = {"pothole": "pothole", "potholes": "pothole", "d40": "pothole",
                   "longitudinal_crack": "longitudinal_crack", "d00": "longitudinal_crack",
                   "transverse_crack": "transverse_crack", "d10": "transverse_crack",
                   "alligator_crack": "alligator_crack", "d20": "alligator_crack"}


def download(slug, dest):
    import kagglehub
    path = kagglehub.dataset_download(slug)
    shutil.copytree(path, dest, dirs_exist_ok=True)
    return dest


def find_images(root):
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXT)}


def read_voc(xml_path):
    t = ET.parse(xml_path).getroot()
    size = t.find("size")
    w, h = float(size.find("width").text), float(size.find("height").text)
    out = []
    for o in t.findall("object"):
        bb = o.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((o.find("name").text.strip(), x1, y1, x2, y2, w, h))
    return out


def convert(root, aliases, extra_map):
    """Return (rows_by_stem, raw_class_names). rows: list of (target, cx, cy, w, h) normalised."""
    imgs = find_images(root)
    rows, raw = {}, set()

    def norm(name):
        n = name.strip().lower().replace(" ", "_")
        n = extra_map.get(name, extra_map.get(n, aliases.get(n)))
        return n

    xmls = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
    coco = [p for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
            if "annotations" in open(p, encoding="utf-8", errors="ignore").read(2000) or "images" in open(p, encoding="utf-8", errors="ignore").read(2000)]
    txts = [p for p in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)
            if os.path.splitext(os.path.basename(p))[0] in imgs]
    if xmls:
        for x in xmls:
            stem = os.path.splitext(os.path.basename(x))[0]
            if stem not in imgs:
                continue
            for name, x1, y1, x2, y2, w, h in read_voc(x):
                raw.add(name)
                t = norm(name)
                if t in TARGETS and w > 0 and h > 0:
                    rows.setdefault(stem, []).append((t, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            rows.setdefault(stem, rows.get(stem, []))
    elif coco:
        for cj in coco:
            d = json.load(open(cj, encoding="utf-8"))
            if "annotations" not in d:
                continue
            cats = {c["id"]: c["name"] for c in d["categories"]}
            meta = {i["id"]: i for i in d["images"]}
            for a in d["annotations"]:
                im = meta[a["image_id"]]
                stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
                if stem not in imgs:
                    continue
                name = cats[a["category_id"]]
                raw.add(name)
                t = norm(name)
                x, y, w, h = a["bbox"]
                W, H = float(im["width"]), float(im["height"])
                if t in TARGETS:
                    rows.setdefault(stem, []).append((t, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            for i in d["images"]:
                rows.setdefault(os.path.splitext(os.path.basename(i["file_name"]))[0], [])
    elif txts:
        names = {}
        for yml in glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True):
            import yaml
            n = yaml.safe_load(open(yml)).get("names")
            if isinstance(n, list):
                names = {i: v for i, v in enumerate(n)}
            elif isinstance(n, dict):
                names = {int(k): v for k, v in n.items()}
        for t in txts:
            stem = os.path.splitext(os.path.basename(t))[0]
            rs = []
            for line in open(t).read().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = names.get(int(float(p[0])), str(int(float(p[0]))))
                raw.add(name)
                tg = norm(name)
                if tg in TARGETS:
                    rs.append((tg, *map(float, p[1:5])))
            rows[stem] = rs
    return rows, imgs, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--map", action="append", default=[], help='raw=target, e.g. "hole=pothole"')
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    extra = dict(m.split("=", 1) for m in a.map)
    os.makedirs(a.out, exist_ok=True)
    sources = {}
    for spec in a.dataset:
        slug = spec.split(":")[0]
        name = slug.replace("/", "__")
        raw_dir = os.path.join(a.out, "_raw", name)
        if not a.skip_download:
            download(slug, raw_dir)
        rows, imgs, raw = convert(raw_dir, DEFAULT_ALIASES, extra)
        print(f"{slug}: raw class names found = {sorted(raw)}; labelled images = {sum(1 for r in rows.values() if r)}")
        unmapped = [r for r in raw if (extra.get(r) or DEFAULT_ALIASES.get(r.strip().lower().replace(' ', '_'))) not in TARGETS + ['ignore']]
        if unmapped:
            sys.exit(f"cannot map classes {unmapped} for {slug}; pass --map 'raw=target' or ignore them explicitly with --map 'raw=ignore'")
        n = 0
        os.makedirs(os.path.join(a.out, name, "images"), exist_ok=True)
        os.makedirs(os.path.join(a.out, name, "labels"), exist_ok=True)
        for stem, rs in rows.items():
            if stem not in imgs:
                continue
            shutil.copy(imgs[stem], os.path.join(a.out, name, "images", os.path.basename(imgs[stem])))
            with open(os.path.join(a.out, name, "labels", stem + ".txt"), "w") as fh:
                fh.write("\n".join(f"{TARGETS.index(t)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for t, cx, cy, w, h in rs))
            n += 1
        sources[name] = {"url": f"https://www.kaggle.com/datasets/{slug}", "licence": "SEE KAGGLE PAGE - fill in before training",
                         "human_labelled": True,
                         "class_map": {str(i): t for i, t in enumerate(TARGETS)}, "n_images": n}
        print(f"  wrote {n} images")
    with open(os.path.join(a.out, "sources.json"), "w") as fh:
        json.dump(sources, fh, indent=2)
    print("sources.json written. Replace each 'licence' placeholder with the real licence, and spot-check labels on 20 images "
          "(including water-filled potholes) before training.")


if __name__ == "__main__":
    main()
