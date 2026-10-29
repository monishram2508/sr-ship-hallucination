"""Phase 2 (GPU half) and Phase 3: build the three training sets and train the ship-only detectors.

Imported by notebooks/train.ipynb on Kaggle or Colab, and by a local smoke test. Expects the upload
layout written by scripts/03_package_for_kaggle.py:

    <upload>/code/{scripts,v2,weights}/   <upload>/data/{images,labels,check_sr,trainset_manifest.csv}

prepare()      HR tiles -> v1's degrade() -> LR (hash-checked against the local manifest) ->
               bicubic and v1's run_sr() -> work/{hr,bicubic,sr}/{images,labels}/{train,val}/
time_epochs()  a short run with the real settings, to project the full budget before launching it
train_one()    one condition x seed, saving last.pt, results.csv, args.yaml, run_info.json, and the
               data yaml + manifest it used
"""
import csv, json, platform, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "v2"))
from common_v2 import CFG, array_md5, load_v1_script  # noqa: E402

T = CFG["train"]


def _split_rows(rows, limit):
    """For smoke tests: the first `limit` train tiles plus a few val tiles, so both splits exist."""
    if not limit:
        return rows
    tr = [r for r in rows if r["split"] == "train"][:limit]
    va = [r for r in rows if r["split"] == "val"][:max(2, limit // 5)]
    return tr + va


def prepare(data_dir, work, device="cuda", limit=0):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from tqdm import tqdm
    data_dir, work = Path(data_dir), Path(work)
    m03 = load_v1_script("03_degrade_sr")
    rows = _split_rows(list(csv.DictReader(open(data_dir / "trainset_manifest.csv"))), limit)
    model = m03.load_sr(device)
    hr_bad, lr_bad, t_sr = [], [], []
    for r in tqdm(rows, desc="degrade + bicubic + SR"):
        t, sp = r["tile"], r["split"]
        src = data_dir / "images" / f"{t}.jpg"
        hr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if array_md5(hr) != r["hr_md5"]:
            hr_bad.append(t)
        lr = m03.degrade(hr)
        if array_md5(lr) != r["lr_md5"]:
            lr_bad.append(t)
        bic = cv2.resize(lr, (CFG["tile"], CFG["tile"]), interpolation=cv2.INTER_CUBIC)
        t0 = time.perf_counter()
        sr = m03.run_sr(model, lr, device)
        t_sr.append(time.perf_counter() - t0)
        for v in ("hr", "bicubic", "sr"):
            (work / v / "images" / sp).mkdir(parents=True, exist_ok=True)
            (work / v / "labels" / sp).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_dir / "labels" / f"{t}.txt", work / v / "labels" / sp / f"{t}.txt")
        shutil.copyfile(src, work / "hr" / "images" / sp / f"{t}.jpg")
        cv2.imwrite(str(work / "bicubic" / "images" / sp / f"{t}.png"), bic)
        cv2.imwrite(str(work / "sr" / "images" / sp / f"{t}.png"), sr)

    check = {}
    for p in sorted((data_dir / "check_sr").glob("*.png")):
        r = next((x for x in rows if x["tile"] == p.stem), None)
        if r is None:
            continue
        here = cv2.imread(str(work / "sr" / "images" / r["split"] / p.name), cv2.IMREAD_COLOR)
        local = cv2.imread(str(p), cv2.IMREAD_COLOR)
        check[p.stem] = float("inf") if np.array_equal(here, local) else float(psnr(local, here, data_range=255))
    rep = {"n_tiles": len(rows), "device": str(device), "hr_hash_mismatch": len(hr_bad),
           "lr_hash_mismatch": len(lr_bad), "lr_mismatch_examples": lr_bad[:10],
           "sr_seconds_per_tile_mean": float(np.mean(t_sr)), "check_sr_psnr_vs_local_db": check}
    (work / "prepare_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


def data_yaml(work, cond):
    import yaml
    src = T["conditions"][cond]["sources"]
    body = {"path": str(Path(work).resolve()), "train": [f"{s}/images/train" for s in src],
            "val": [f"{s}/images/val" for s in src], "names": {0: "ship"}}
    p = Path(work) / f"data_{cond}.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False))
    return p


def _train_args(cond, seed, yaml_path, out_root, name, device, epochs=None, batch=None, imgsz=None,
                close_mosaic=None, workers=None):
    c = T["conditions"][cond]
    args = dict(data=str(yaml_path), imgsz=imgsz or T["imgsz"], epochs=epochs or c["epochs"],
                batch=batch or T["batch"], single_cls=True, flipud=T["flipud"],
                deterministic=T["deterministic"], seed=int(seed), patience=T["patience"],
                warmup_epochs=c["warmup_epochs"],
                close_mosaic=c["close_mosaic"] if close_mosaic is None else close_mosaic,
                cache=T["cache"], device=device, project=str(out_root), name=name, exist_ok=True,
                plots=False, verbose=False)
    if workers is not None:
        args["workers"] = workers
    return args


def _env():
    import torch
    import ultralytics
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": gpu, "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__}


def _epoch_seconds(results_csv):
    """Per-epoch wall time (training + that epoch's validation) from ultralytics' own cumulative
    `time` column. A fit-epoch-end callback is not used: it also fires during the final validation."""
    with open(results_csv) as f:
        t = [float(r["time"]) for r in csv.DictReader(f)]
    return [float(x) for x in np.diff([0.0] + t)]


def train_one(cond, seed, work, out_root, device=0, manifest=None, **override):
    """Train one condition x seed from COCO yolov8n weights (never the DOTA OBB weights)."""
    from ultralytics import YOLO
    name = f"{cond}_s{seed}"
    yml = data_yaml(work, cond)
    model = YOLO(T["model_cfg"]).load(T["init_weights"])
    args = _train_args(cond, seed, yml, out_root, name, device, **override)
    t0 = time.time()
    model.train(**args)
    tr = model.trainer
    save = Path(tr.save_dir)
    opt = tr.optimizer
    info = {"condition": cond, "seed": int(seed), "train_args": args, "seconds": time.time() - t0,
            "epoch_seconds": _epoch_seconds(save / "results.csv"),
            "optimizer": type(opt).__name__, "lr0_resolved": opt.param_groups[0].get("initial_lr"),
            "batch_requested": args["batch"], "batch_used": int(tr.batch_size),
            "accumulate": int(tr.accumulate), "n_train_images": len(tr.train_loader.dataset),
            "batches_per_epoch": len(tr.train_loader), "env": _env()}
    (save / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    shutil.copyfile(yml, save / yml.name)
    if manifest:
        shutil.copyfile(manifest, save / "trainset_manifest.csv")
    if info["batch_used"] != info["batch_requested"]:
        print(f"!! {name}: batch reduced {info['batch_requested']} -> {info['batch_used']} (OOM)")
    return info


def time_epochs(work, out_root, cond="M_SR", n_epochs=2, device=0):
    """Two epochs with the real settings (mosaic on). Epoch 2 excludes caching and warm-up costs."""
    info = train_one(cond, 0, work, out_root, device=device, epochs=n_epochs, close_mosaic=0)
    per = info["epoch_seconds"][-1]
    # an epoch's cost scales with its image count: M_HR has 1 source, M_SR/M_BIC have 2
    n_src = len(T["conditions"][cond]["sources"])
    run_s = {k: c["epochs"] * per * len(c["sources"]) / n_src for k, c in T["conditions"].items()}
    proj = {"measured_condition": cond, "seconds_per_epoch": per,
            "projected_minutes_per_run": {k: round(v / 60, 1) for k, v in run_s.items()},
            "projected_gpu_hours_all_seeds": round(sum(run_s.values()) * len(T["seeds"]) / 3600, 2),
            "note": "epoch time (training + per-epoch validation) scaled by each condition's image "
                    "count; excludes prepare(), which runs once per session"}
    (Path(out_root) / "timing_projection.json").write_text(json.dumps(proj, indent=2))
    print(json.dumps(proj, indent=2))
    return proj


def zip_runs(out_root, names, zip_path, include_weights=True):
    """Only what Phase 4 needs: last.pt, results.csv, args.yaml, run_info.json, data yaml, manifest.
    The timing run is zipped without weights so a 2-epoch model can never be mistaken for a real one."""
    import zipfile
    keep = ("results.csv", "args.yaml", "run_info.json", "trainset_manifest.csv")
    keep = (("weights/last.pt",) if include_weights else ()) + keep
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            d = Path(out_root) / n
            for k in keep:
                if (d / k).exists():
                    z.write(d / k, f"{n}/{k}")
            for y in d.glob("data_*.yaml"):
                z.write(y, f"{n}/{y.name}")
        for extra in ("timing_projection.json",):
            if (Path(out_root) / extra).exists():
                z.write(Path(out_root) / extra, extra)
    print(f"-> {zip_path} ({Path(zip_path).stat().st_size / 2**20:.1f} MB)")
