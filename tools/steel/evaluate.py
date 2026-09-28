"""
Evaluate best.pth of an experiment on its val split (and optionally measure FPS).

    python tools/steel/evaluate.py [experiment.yml] [--ckpt path] [--fps]

Writes <run>/val_predictions.json (COCO results) and <run>/eval.json; prints mAP@0.5, mAP@0.5:0.95,
precision / recall / F1 (best-F1 point, IoU 0.5) and AP@0.5 per class.
--fps: end-to-end speed like the project's evaluate_model - batch 1, per image read + resize + model + postprocess.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig  # noqa: E402
from engine.misc.metrics_log import pr_at_best_f1  # noqa: E402


class Deployed(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_sizes):
        return self.postprocessor(self.model(images), orig_sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default=str(REPO / "experiment.yml"))
    ap.add_argument("--ckpt", default=None, help="default: <run>/best.pth")
    ap.add_argument("--fps", action="store_true")
    a = ap.parse_args()
    exp = yaml.safe_load(Path(a.experiment).read_text(encoding="utf-8"))
    cfg_path = REPO / "configs" / "_generated" / f"{exp['name']}.yml"
    cfg = YAMLConfig(str(cfg_path))
    run_dir = Path(cfg.yaml_cfg["output_dir"])
    ckpt = Path(a.ckpt) if a.ckpt else run_dir / "best.pth"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
    model = Deployed(cfg).cuda().eval()

    h, w = cfg.yaml_cfg["eval_spatial_size"]
    ds = cfg.yaml_cfg["val_dataloader"]["dataset"]
    img_dir, ann = Path(ds["img_folder"]), Path(ds["ann_file"])
    tf = T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    images = json.loads(ann.read_text())["images"]

    res = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = images[i:i + 16]
            pil = [Image.open(img_dir / im["file_name"]).convert("RGB") for im in batch]
            labels, boxes, scores = model(torch.stack([tf(p) for p in pil]).cuda(),
                                          torch.tensor([[p.width, p.height] for p in pil]).cuda())
            for im, lab, box, sc in zip(batch, labels, boxes, scores):
                for (x1, y1, x2, y2), c, s in zip(box.tolist(), lab.tolist(), sc.tolist()):
                    res.append({"image_id": im["id"], "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(s)})
    (run_dir / "val_predictions.json").write_text(json.dumps(res))

    from faster_coco_eval import COCO, COCOeval_faster

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))
        e = COCOeval_faster(gt, gt.loadRes(res), iouType="bbox", print_function=print, separate_eval=True)
        e.evaluate()
        e.accumulate()
        e.summarize()
    p, r, f1, ap50 = pr_at_best_f1(e)
    names = [c["name"] for c in gt.loadCats(sorted(gt.getCatIds()))]
    out = {"checkpoint": str(ckpt), "mAP50": float(e.stats[1]), "mAP50_95": float(e.stats[0]), "precision": p,
           "recall": r, "f1": f1, "AP50_per_class": dict(zip(names, ap50))}

    if a.fps:
        def one(img_path):
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                model(tf(img)[None].cuda(), torch.tensor([[img.width, img.height]], device="cuda"))
            torch.cuda.synchronize()

        paths = [img_dir / im["file_name"] for im in images]
        for pth in paths[:20]:
            one(pth)
        t0 = time.perf_counter()
        for pth in paths[:300]:
            one(pth)
        out["fps"] = 300 / (time.perf_counter() - t0)

    (run_dir / "eval.json").write_text(json.dumps(out, indent=1))
    print(f"checkpoint {ckpt}")
    print(f"mAP@0.5 {out['mAP50']:.4f} | mAP@0.5:0.95 {out['mAP50_95']:.4f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    print("AP@0.5 per class: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(names, ap50)))
    if a.fps:
        print(f"FPS (batch 1, FP32, read+resize+model+postprocess): {out['fps']:.1f}")


if __name__ == "__main__":
    main()
