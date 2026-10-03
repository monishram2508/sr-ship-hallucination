"""Phase 2 — degrade HR tiles to LR, then restore two ways (bicubic, Real-ESRGAN).

HR 1024 --blur(sigma=2)+INTER_AREA 4x--> LR 256 --+--> INTER_CUBIC  --> bicubic 1024
                                                  +--> Real-ESRGAN  --> sr 1024

Writes data/versions/{hr,lr,bicubic,sr}/{setA,setB}/images/*.png
plus a copy of the labels into every version, and the per-version YAMLs.
"""
import argparse, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, SR_WEIGHTS, TILE, LR, SCALE, BLUR_SIGMA, FIGURES, device  # noqa: E402

VERSIONS = ("hr", "lr", "bicubic", "sr")


def degrade(hr):
    """HR uint8 BGR 1024 -> LR uint8 BGR 256. Shermeyer & Van Etten (2019) sensor model:
    Gaussian PSF blur then inter-area decimation."""
    blurred = cv2.GaussianBlur(hr, (0, 0), sigmaX=BLUR_SIGMA)
    return cv2.resize(blurred, (LR, LR), interpolation=cv2.INTER_AREA)


def load_sr(dev):
    from spandrel import ModelLoader
    desc = ModelLoader().load_from_file(str(SR_WEIGHTS))
    assert desc.scale == SCALE, f"expected x{SCALE}, got x{desc.scale}"
    return desc.to(dev).eval()


def run_sr(model, lr_bgr, dev):
    x = torch.from_numpy(lr_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(dev)
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    out = (y[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).round().astype("uint8")
    del x, y
    return out


def class_names():
    """Read DOTAv1 class names straight from the installed ultralytics cfg (no hand-typing)."""
    import yaml
    from ultralytics.utils import ROOT as U_ROOT
    cfg = yaml.safe_load((Path(U_ROOT) / "cfg/datasets/DOTAv1.yaml").read_text())
    return cfg["names"]


def write_yamls(setname, names):
    ydir = DATA / "yaml"
    ydir.mkdir(parents=True, exist_ok=True)
    import yaml
    for v in VERSIONS:
        p = ydir / f"{v}_{setname}.yaml"
        body = {
            "path": str((DATA / "versions" / v / setname).resolve()),
            "train": "images",
            "val": "images",
            "names": names,
        }
        p.write_text(yaml.safe_dump(body, sort_keys=False))
    return ydir


def sanity_panel(tile_id, hr, bic, sr, out):
    gap = 8
    h = TILE
    canvas = np.full((h, TILE * 3 + gap * 2, 3), 255, np.uint8)
    for i, img in enumerate((hr, bic, sr)):
        x0 = i * (TILE + gap)
        canvas[:, x0:x0 + TILE] = img
    for i, lbl in enumerate(("HR (ground truth)", "bicubic x4", "Real-ESRGAN x4")):
        x0 = i * (TILE + gap)
        cv2.rectangle(canvas, (x0, 0), (x0 + 430, 46), (0, 0, 0), -1)
        cv2.putText(canvas, lbl, (x0 + 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(out), cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="setname", required=True, choices=["setA", "setB"])
    ap.add_argument("--limit", type=int, default=0, help="process only the first N tiles (timing runs)")
    ap.add_argument("--sanity", type=int, default=0, help="write N 3-panel sanity figures")
    args = ap.parse_args()

    ids = [l.strip() for l in (DATA / "sets" / f"{args.setname}.txt").read_text().splitlines() if l.strip()]
    if args.limit:
        ids = ids[: args.limit]
    src_img = DATA / "tiles" / "images"
    src_lbl = DATA / "tiles" / "labels"

    for v in VERSIONS:
        (DATA / "versions" / v / args.setname / "images").mkdir(parents=True, exist_ok=True)
        (DATA / "versions" / v / args.setname / "labels").mkdir(parents=True, exist_ok=True)

    dev = device()
    model = load_sr(dev)
    print(f"device={dev} tiles={len(ids)} sigma={BLUR_SIGMA}")

    times = []
    for k, tid in enumerate(tqdm(ids, desc=f"{args.setname} degrade+SR")):
        hr_path = next(src_img.glob(f"{tid}.*"))
        hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        assert hr.shape[:2] == (TILE, TILE), f"{tid} is {hr.shape[:2]}, expected {TILE}x{TILE}"

        lr = degrade(hr)
        bic = cv2.resize(lr, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        t0 = time.perf_counter()
        sr = run_sr(model, lr, dev)
        times.append(time.perf_counter() - t0)

        for v, img in (("hr", hr), ("lr", lr), ("bicubic", bic), ("sr", sr)):
            cv2.imwrite(str(DATA / "versions" / v / args.setname / "images" / f"{tid}.png"), img)
            lp = src_lbl / f"{tid}.txt"
            dst = DATA / "versions" / v / args.setname / "labels" / f"{tid}.txt"
            if lp.exists():
                shutil.copyfile(lp, dst)
            else:
                dst.write_text("")

        if k < args.sanity:
            FIGURES.mkdir(parents=True, exist_ok=True)
            sanity_panel(tid, hr, bic, sr, FIGURES / f"sanity_{args.setname}_{tid}.png")

    t = np.array(times)
    print(f"SR per tile: mean {t.mean():.2f}s  median {np.median(t):.2f}s  max {t.max():.2f}s  total {t.sum()/60:.1f}min")

    names = class_names()
    ydir = write_yamls(args.setname, names)
    print(f"yamls -> {ydir}")
    for v in VERSIONS:
        n = len(list((DATA / "versions" / v / args.setname / "images").glob("*.png")))
        m = len(list((DATA / "versions" / v / args.setname / "labels").glob("*.txt")))
        print(f"  {v:8s} images={n:4d} labels={m:4d}")


if __name__ == "__main__":
    main()
