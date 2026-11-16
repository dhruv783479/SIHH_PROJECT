"""Export examples/annotated_samples as an extra training source (pothole boxes only).

    python training/export_samples_as_source.py --data-root /content/data

Writes <data_root>/annotated_samples/{images,labels} and merges an entry into <data_root>/sources.json
so training/train_detectors_real.py picks it up. Only the hand-placed pothole boxes are exported
(cars/persons/zebra are not target classes of that trainer). The images go through the same
duplicate-clustered split as everything else, so they can land in train, val or test; they are
never forced into test and never used to claim a metric on their own.
"""
import argparse, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "examples", "annotated_samples")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); a = ap.parse_args()
    ann = json.load(open(os.path.join(SRC, "annotations.json")))
    out = os.path.join(a.data_root, "annotated_samples")
    os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    n = 0
    for name, d in ann["images"].items():
        pots = [b for b in d["boxes"] if b[0] == "pothole"]
        if not pots:
            continue  # no target-class labels -> skip rather than teach 'no pothole' on a pothole-free-labelled image
        W, H = d["size"]
        shutil.copy(os.path.join(SRC, "images", name + ".png"), os.path.join(out, "images", name + ".png"))
        with open(os.path.join(out, "labels", name + ".txt"), "w") as fh:
            for _, x1, y1, x2, y2, _src in pots:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                fh.write(f"3 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        n += 1
    sp = os.path.join(a.data_root, "sources.json")
    src = json.load(open(sp)) if os.path.exists(sp) else {}
    src["annotated_samples"] = {"url": "local: examples/annotated_samples (hand-placed boxes, stock photos)",
                                "licence": "CHECK redistribution rights of the stock photos before publishing",
                                "human_labelled": True,
                                "class_map": {"3": "pothole"}, "n_images": n}
    json.dump(src, open(sp, "w"), indent=2)
    print(f"exported {n} images to {out}; sources.json updated")


if __name__ == "__main__":
    main()
