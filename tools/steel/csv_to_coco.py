"""
Convert a box CSV (ImageId, ClassId, x_min, y_min, x_max, y_max, split) to COCO json, one file per split.

Images are not copied: file_name is the ImageId, DEIMv2 reads them from the images folder given in the config.
Class ids are optionally merged (e.g. {6: 1, 5: 3}) and then remapped to contiguous category ids 0..K-1
(what DEIMv2 expects with remap_mscoco_category: False).

python tools/steel/csv_to_coco.py train_bboxes.csv images_dir out_dir [--merge 6:1,5:3]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def convert(csv_path, images_dir, out_dir, class_merge=None, splits=("train", "val")):
    class_merge = {int(k): int(v) for k, v in (class_merge or {}).items()}
    rows = list(csv.DictReader(open(csv_path, newline="")))
    for r in rows:
        r["cls"] = class_merge.get(int(r["ClassId"]), int(r["ClassId"]))
    classes = sorted({r["cls"] for r in rows})
    cat_id = {c: i for i, c in enumerate(classes)}
    categories = [{"id": cat_id[c], "name": f"defect_{c}", "supercategory": "defect"} for c in classes]

    by_split = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_split[r["split"]][r["ImageId"]].append(r)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in splits:
        images, anns = [], []
        for i, (image_id, boxes) in enumerate(sorted(by_split[split].items()), start=1):
            with Image.open(Path(images_dir) / image_id) as im:
                w, h = im.size
            images.append({"id": i, "file_name": image_id, "width": w, "height": h})
            for r in boxes:
                x0, y0, x1, y1 = (float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat_id[r["cls"]],
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps({"images": images, "annotations": anns, "categories": categories}))
        written[split] = path
        print(f"{split}: {len(images)} images, {len(anns)} boxes -> {path}")
    return written, categories


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("images")
    ap.add_argument("out_dir")
    ap.add_argument("--merge", default="", help="old:new pairs, e.g. 6:1,5:3")
    a = ap.parse_args()
    merge = dict(p.split(":") for p in a.merge.split(",") if p)
    convert(a.csv, a.images, a.out_dir, merge)
