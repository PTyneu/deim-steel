"""
Download every weight file an experiment needs.

    python tools/steel/download_weights.py [s|m|l|x|all ...] [--no-backbone] [--no-detector]

Without a model argument the model of experiment.yml is used (x if there is no experiment.yml).
  detector  weights/deimv2_dinov3_<m>_coco.pth   DEIMv2 COCO checkpoint - the fine-tuning start point (train.py -t)
  backbone  ckpts/...                            names expected by the upstream configs (configs/deimv2/*.yml):
            s, m: distilled ViT-Tiny / ViT-Tiny+ (Google Drive, DEIMv2 authors)
            l, x: DINOv3 ViT-S16 / ViT-S16+ - timm's re-hosted weights (Hugging Face, no gated access) converted
                  to the official DINOv3 key format and checked against timm (identical outputs)
The backbone files are only read when training starts from a bare backbone (experiment.yml: weights: none) or with
the upstream configs; fine-tuning from the detector checkpoint does not need them.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DETECTOR_GDRIVE = {  # README model zoo, "checkpoint" column
    "s": "1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL",
    "m": "1nPKDHrotusQ748O1cQXJfi5wdShq6bKp",
    "l": "1dRJfVHr9HtpdvaHlnQP460yPVHynMray",
    "x": "1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7",
}
BACKBONE_GDRIVE = {"s": ("vitt_distill.pt", "1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs"),
                   "m": ("vittplus_distill.pt", "1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt")}
BACKBONE_DINOV3 = {"l": ("dinov3_vits16_pretrain_lvd1689m-08c60483.pth", "dinov3_vits16"),
                   "x": ("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", "dinov3_vits16plus")}


def gdrive(file_id, out):
    import gdown

    if out.exists() and out.stat().st_size > 0:
        print(f"ok (exists)  {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    got = gdown.download(id=file_id, output=str(out), quiet=False)
    if not got or not out.exists():
        sys.exit(f"download failed: https://drive.google.com/file/d/{file_id}  -> save it manually as {out}")


def check_detector(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in state or "ema" in state, f"{path} is not a DEIMv2 checkpoint"
    print(f"ok           {path} ({path.stat().st_size / 2**20:.0f} MB)")


def dinov3_backbone(file_name, arch):
    import torch

    from engine.backbone.dinov3.vision_transformer import DinoVisionTransformer
    from tools.steel.convert_dinov3_timm import TIMM_NAMES, convert

    out = REPO / "ckpts" / file_name
    if out.exists() and out.stat().st_size > 0:
        print(f"ok (exists)  {out}")
        return
    import timm

    tm = timm.create_model(TIMM_NAMES[arch], pretrained=True).eval()
    dm = DinoVisionTransformer(name=arch).eval()
    sd = convert(tm.state_dict(), dm.state_dict())
    dm.load_state_dict(sd)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        a = tm.forward_features(x)[:, tm.num_prefix_tokens:]
        o = dm.forward_features(x)
        b = (o[0] if isinstance(o, list) else o)["x_norm_patchtokens"]
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, f"conversion check failed (max diff {diff})"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"ok           {out} (converted from timm {TIMM_NAMES[arch]}, max diff {diff:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="s m l x | all (default: model of experiment.yml)")
    ap.add_argument("--no-backbone", action="store_true")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--backbone", action="store_true", help="(default) kept for readability")
    a = ap.parse_args()
    models = a.models
    if not models:
        exp = REPO / "experiment.yml"
        if exp.exists():
            import yaml

            models = [str(yaml.safe_load(exp.read_text(encoding="utf-8")).get("model", "x")).lower()]
        else:
            models = ["x"]
    if "all" in models:
        models = ["s", "m", "l", "x"]
    for m in models:
        assert m in DETECTOR_GDRIVE, f"unknown model {m}"
        print(f"--- DEIMv2-{m.upper()}")
        if not a.no_detector:
            out = REPO / "weights" / f"deimv2_dinov3_{m}_coco.pth"
            gdrive(DETECTOR_GDRIVE[m], out)
            check_detector(out)
        if not a.no_backbone:
            if m in BACKBONE_GDRIVE:
                name, fid = BACKBONE_GDRIVE[m]
                gdrive(fid, REPO / "ckpts" / name)
            else:
                dinov3_backbone(*BACKBONE_DINOV3[m])


if __name__ == "__main__":
    main()
