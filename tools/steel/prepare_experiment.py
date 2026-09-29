"""
Build a complete DEIMv2 training config from the single editable file (experiment.yml) and optionally train.

    python tools/steel/prepare_experiment.py experiment.yml           # -> configs/_generated/<name>.yml + command
    python tools/steel/prepare_experiment.py experiment.yml --train   # ... and run train.py with it
                                                                     #     (several `devices`: DDP via torchrun)

What is derived from experiment.yml (everything else comes from the chosen model's COCO recipe,
configs/deimv2/deimv2_dinov3_<model>_coco.yml, which stays untouched - the upstream way of training still works):
  * data: CSV (converted to COCO json once, cached in data_cache/) or ready COCO json + image folders;
    category ids are remapped to 0..K-1 when needed; num_classes is read from the annotations
  * input size [h, w]: Resize in train/val, eval_spatial_size; rectangular inputs drop Mosaic and multi-scale
    batches (both build square canvases)
  * the recipe's schedule (augmentation stages, flat-cosine lr, no-aug tail, matcher switch) scaled to `epochs`
  * optimizer: AdamW (recipe lr scaled linearly to the batch) or SGD (lr 0.01, backbone x0.02, Nesterov, clip 10)
  * run folder outputs/<name>/: best.pth only + metrics.csv / metrics.png / log.txt (see engine/solver/det_solver.py)
  * devices: GPU ids; more than one -> DDP on one machine (torchrun). batch_size stays the total over all GPUs
    (DEIM's total_batch_size), so iterations per epoch, lr and warmups do not depend on the number of GPUs
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from engine.core.yaml_utils import load_config  # noqa: E402

MODEL_CFG = {m: f"deimv2_dinov3_{m}_coco.yml" for m in ("s", "m", "l", "x")}
BACKBONE_CKPT = {  # backbone-only init (weights: none); fetched by scripts/download_weights.sh --backbone
    "s": "ckpts/vitt_distill.pt",
    "m": "ckpts/vittplus_distill.pt",
    "l": "ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "x": "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
}


def path_of(p, base):
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return (p if p.is_absolute() else (base / p)).resolve()


def auto(v):
    return v is None or (isinstance(v, str) and v.lower() == "auto")


def gpu_ids(v):
    """devices: 0 | [0, 1] | "0,1" -> ['0', '1']; absent -> None (CUDA_VISIBLE_DEVICES is left as it is, one GPU)."""
    if v is None:
        return None
    ids = [str(int(x)) for x in v] if isinstance(v, (list, tuple)) else [s.strip() for s in str(v).split(",")]
    return [i for i in ids if i] or None


def contiguous_ann(ann, cache_dir):
    """DEIMv2 (remap_mscoco_category: False) needs category ids 0..K-1; write a remapped copy otherwise."""
    d = json.loads(Path(ann).read_text())
    ids = sorted(c["id"] for c in d["categories"])
    if ids == list(range(len(ids))):
        return Path(ann), len(ids), len(d["images"])
    remap = {old: new for new, old in enumerate(ids)}
    for c in d["categories"]:
        c["id"] = remap[c["id"]]
    for a in d["annotations"]:
        a["category_id"] = remap[a["category_id"]]
    out = cache_dir / (Path(ann).stem + "_ids0.json")
    out.write_text(json.dumps(d))
    return out, len(ids), len(d["images"])


def prepare_data(data, base, name):
    cache = REPO / "data_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    test = None
    if data.get("format", "coco") == "csv":
        from tools.steel.csv_to_coco import convert

        csv_path = path_of(data["csv"], base)
        images = path_of(data["images"], base) if data.get("images") else None  # legacy ImageId / relative paths
        merge = {str(k): str(v) for k, v in (data.get("class_merge") or {}).items()}
        key = hashlib.md5(f"v2|{csv_path}|{os.path.getmtime(csv_path)}|{images}|{sorted(merge.items())}".encode())
        stamp = cache / f"source_{key.hexdigest()[:10]}.txt"
        if not stamp.exists():
            for old in list(cache.glob("*.json")) + list(cache.glob("source_*.txt")):
                old.unlink()  # stale splits of an earlier conversion
            convert(csv_path, cache, images, merge)
            stamp.write_text(f"{csv_path}\n{images}\n{merge}\n")
        root = images or csv_path.parent  # file_name is an absolute path, it wins over the folder
        train_img = val_img = root
        train_ann, val_ann = cache / "train.json", cache / "val.json"
        if (cache / "test.json").exists():
            test = (root, cache / "test.json")
    else:
        train_img, val_img = path_of(data["train_images"], base), path_of(data["val_images"], base)
        train_ann, val_ann = path_of(data["train_ann"], base), path_of(data["val_ann"], base)
        if data.get("test_ann"):
            test = (path_of(data["test_images"], base), path_of(data["test_ann"], base))
    train_ann, k_train, n_train = contiguous_ann(train_ann, cache)
    val_ann, k_val, _ = contiguous_ann(val_ann, cache)
    assert k_train == k_val, f"train has {k_train} classes, val has {k_val}"
    if test:
        test = (test[0], contiguous_ann(test[1], cache)[0])
    return train_img, train_ann, val_img, val_ann, k_train, n_train, test


def build(exp, base):
    name, model = exp["name"], str(exp.get("model", "x")).lower()
    assert model in MODEL_CFG, f"model must be one of {list(MODEL_CFG)}"
    recipe = load_config(str(REPO / "configs" / "deimv2" / MODEL_CFG[model]), {})
    r_epochs, r_policy = recipe["epoches"], recipe["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"]
    r_batch = recipe["train_dataloader"]["total_batch_size"]

    train_img, train_ann, val_img, val_ann, num_classes, n_train, test = prepare_data(exp["data"], base, name)
    E, batch, val_batch = int(exp.get("epochs", 12)), int(exp.get("batch_size", 16)), int(exp.get("val_batch_size", 16))
    n_gpu = len(gpu_ids(exp.get("devices")) or [0])
    if batch % n_gpu or val_batch % n_gpu:
        raise ValueError(f"batch_size ({batch}) and val_batch_size ({val_batch}) are totals over all GPUs: "
                         f"make them divisible by the number of devices ({n_gpu})")
    h, w = (int(v) for v in exp.get("input_size", [640, 640]))
    assert h % 32 == 0 and w % 32 == 0, "input_size sides must be multiples of 32"
    square = h == w
    mosaic = square if auto(exp.get("mosaic")) else bool(exp.get("mosaic"))
    multiscale = bool(exp.get("multiscale", False))
    if not square and (mosaic or multiscale):
        raise ValueError("Mosaic / multi-scale build square canvases; use them only with a square input_size")

    # schedule: recipe proportions scaled to E epochs
    a, b, c = r_policy
    # clean (no-aug) tail; runs shorter than 3 epochs have none, otherwise the stage switch would fall on epoch 0
    noaug = max(1, round(E * (r_epochs - c) / r_epochs)) if E >= 3 else 0
    p2 = E - noaug
    p0 = min(round(E * a / r_epochs), p2)
    p1 = min(p0 + max(1, round((p2 - p0) * (b - a) / max(c - a, 1))), p2)
    flat = max(1, round(E * recipe["flat_epoch"] / r_epochs))
    mce = max(1, round(E * recipe["DEIMCriterion"]["matcher"]["matcher_change_epoch"] / r_epochs))
    iters = max(1, n_train // batch)
    warmup_iter, ema_warmups = max(50, round(0.5 * iters)), max(100, round(1.5 * iters))

    # optimizer
    opt = str(exp.get("optimizer", "adamw")).lower()
    groups = [dict(g) for g in recipe["optimizer"]["params"]]
    if opt == "sgd":
        lr = 0.01 if auto(exp.get("lr")) else float(exp["lr"])
        ratio = 0.02 if auto(exp.get("backbone_lr_ratio")) else float(exp["backbone_lr_ratio"])
        wd = 1e-4 if auto(exp.get("weight_decay")) else float(exp["weight_decay"])
    elif opt == "adamw":
        r_lr, r_bb = recipe["optimizer"]["lr"], groups[0]["lr"]
        lr = r_lr * batch / r_batch if auto(exp.get("lr")) else float(exp["lr"])
        ratio = r_bb / r_lr if auto(exp.get("backbone_lr_ratio")) else float(exp["backbone_lr_ratio"])
        wd = recipe["optimizer"]["weight_decay"] if auto(exp.get("weight_decay")) else float(exp["weight_decay"])
    else:
        raise ValueError("optimizer must be adamw or sgd")
    for g in groups:
        if "lr" in g:  # the backbone groups
            g["lr"] = lr * ratio
    optimizer = {"type": "AdamW", "params": groups, "lr": lr, "betas": [0.9, 0.999], "weight_decay": wd}
    if opt == "sgd":  # SGDIgnoreBetas tolerates the AdamW `betas` key merged in from the base configs
        optimizer = {"type": "SGDIgnoreBetas", "params": groups, "lr": lr, "momentum": 0.9, "nesterov": True,
                     "weight_decay": wd}

    # transforms: recipe ops with the requested size; Mosaic only for square inputs
    train_ops = []
    for op in recipe["train_dataloader"]["dataset"]["transforms"]["ops"]:
        op = dict(op)
        if op["type"] == "Mosaic":
            if not mosaic:
                continue
            op["output_size"] = h // 2
        if op["type"] == "Resize":
            op["size"] = [h, w]
        train_ops.append(op)
    val_ops = [dict(op, size=[h, w]) if op["type"] == "Resize" else dict(op)
               for op in recipe["val_dataloader"]["dataset"]["transforms"]["ops"]]

    weights = exp.get("weights", "auto")
    if auto(weights):
        weights = REPO / "weights" / f"deimv2_dinov3_{model}_coco.pth"
    elif str(weights).lower() == "none":
        weights = None
    else:
        weights = path_of(weights, base)
    backbone = None if weights else str(REPO / BACKBONE_CKPT[model])

    run_dir = path_of(exp.get("output_root", "outputs"), base) / name
    cfg = {
        "__include__": [f"../deimv2/{MODEL_CFG[model]}"],
        "output_dir": run_dir.as_posix(),
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "eval_spatial_size": [h, w],
        "epoches": E,
        "flat_epoch": flat,
        "no_aug_epoch": noaug,
        "warmup_iter": warmup_iter,
        "ema": {"warmups": ema_warmups},
        "print_freq": 100,
        "checkpoint_freq": 10 ** 6,
        "save_best_only": bool(exp.get("save_best_only", True)),
        "best_metric": str(exp.get("best_metric", "map50")),
        "plot_metrics": True,
        "DEIMCriterion": {"matcher": {"matcher_change_epoch": mce}},
        "DINOv3STAs": {"weights_path": backbone},
        "optimizer": optimizer,
        "train_dataloader": {
            "total_batch_size": batch,
            "num_workers": int(exp.get("workers", 4)),
            "dataset": {"img_folder": Path(train_img).as_posix(), "ann_file": Path(train_ann).as_posix(),
                        "transforms": {"ops": train_ops, "policy": {"epoch": [p0, p1, p2]},
                                       "mosaic_prob": recipe["train_dataloader"]["dataset"]["transforms"].get(
                                           "mosaic_prob", 0.5) if mosaic else 0.0}},
            "collate_fn": {"mixup_epochs": [p0, p1], "copyblend_epochs": [p0, p2], "stop_epoch": p2,
                           "base_size": h, "base_size_repeat": (recipe["train_dataloader"]["collate_fn"].get(
                               "base_size_repeat") if multiscale else None)},
        },
        "val_dataloader": {
            "total_batch_size": val_batch,
            "num_workers": int(exp.get("workers", 4)),
            "dataset": {"img_folder": Path(val_img).as_posix(), "ann_file": Path(val_ann).as_posix(),
                        "transforms": {"ops": val_ops}},
        },
    }
    if opt == "sgd":
        cfg["clip_max_norm"] = 10.0  # 0.1 of the AdamW recipe would stall plain SGD
    if test:  # not used by training; tools/steel/evaluate.py --split test
        cfg["test_images"], cfg["test_ann"] = Path(test[0]).as_posix(), Path(test[1]).as_posix()
    summary = (f"model DEIMv2-{model.upper()} | input {h}x{w} | {E} epochs, batch {batch}, {opt} lr {lr:.3g} "
               f"(backbone x{ratio:.3g}), wd {wd:.3g} | classes {num_classes}, train images {n_train} | "
               f"aug stages {[p0, p1, p2]}, mosaic {mosaic}, multiscale {multiscale} | test split {bool(test)}"
               + (f" | DDP on {n_gpu} GPUs, batch {batch // n_gpu} per GPU" if n_gpu > 1 else ""))
    return cfg, weights, run_dir, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default=str(REPO / "experiment.yml"))
    ap.add_argument("--train", action="store_true", help="start train.py after writing the config")
    args = ap.parse_args()
    exp_path = Path(args.experiment).resolve()
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    cfg, weights, run_dir, summary = build(exp, exp_path.parent)

    out = REPO / "configs" / "_generated" / f"{exp['name']}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# generated by tools/steel/prepare_experiment.py from {exp_path.name} - edit that file instead\n"
                   + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model = exp.get("model", "x")
    if weights is not None and not Path(weights).exists():
        sys.exit(f"weights not found: {weights}\nrun: bash scripts/download_weights.sh {model}")
    if weights is not None and Path(weights).read_bytes()[:64].startswith(b"version https://git-lfs"):
        sys.exit(f"{weights} is a Git LFS pointer, not the weights.\n"
                 "run: git lfs install && git lfs pull   (or: bash scripts/download_weights.sh)")
    backbone = cfg["DINOv3STAs"]["weights_path"]
    if backbone and not Path(backbone).exists():  # DEIMv2 would silently start the backbone from scratch
        sys.exit(f"backbone weights not found: {backbone}\nrun: bash scripts/download_weights.sh {model} --backbone")

    train = ["train.py", "-c", out.relative_to(REPO).as_posix(), "--seed", str(exp.get("seed", 0))]
    if exp.get("amp", True):
        train.append("--use-amp")
    if weights is not None:
        train += ["-t", Path(weights).as_posix()]
    ids = gpu_ids(exp.get("devices"))
    ddp = bool(ids) and len(ids) > 1
    if ddp:  # one process per GPU; --standalone picks a free port for the rendezvous
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(ids)}"] + train
    else:
        cmd = [sys.executable, "-u"] + train
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if ids:  # ids as nvidia-smi prints them
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(ids))
    print(summary)
    print(f"config:  {out}")
    print(f"results: {run_dir}  (best.pth, metrics.csv, metrics.png, log.txt)")
    print("command: " + (f"CUDA_VISIBLE_DEVICES={','.join(ids)} " if ids else "") + " ".join(cmd))
    if args.train:
        if ddp and os.name == "nt":
            sys.exit("DDP: torchrun from the Windows builds of torch cannot start its store (they lack libuv); "
                     "train on several GPUs under Linux, or set one device")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "train.log", "a", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            sys.exit(proc.wait())


if __name__ == "__main__":
    main()
