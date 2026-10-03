"""Phase 3c — PSNR and SSIM of bicubic and SR against the HR tile."""
import csv, sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

SETS = ("setA", "setB")
VERSIONS = ("bicubic", "sr")


def main():
    rows = []
    for s in SETS:
        hrdir = DATA / "versions" / "hr" / s / "images"
        tiles = sorted(hrdir.glob("*.png"))
        if not tiles:
            print(f"skip {s}: no tiles")
            continue
        for p in tqdm(tiles, desc=f"psnr {s}", leave=False):
            hr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            for v in VERSIONS:
                q = DATA / "versions" / v / s / "images" / p.name
                im = cv2.imread(str(q), cv2.IMREAD_COLOR)
                rows.append({
                    "set": s, "tile": p.stem, "version": v,
                    "psnr": float(psnr(hr, im, data_range=255)),
                    "ssim": float(ssim(hr, im, data_range=255, channel_axis=2)),
                })
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "psnr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "tile", "version", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'set':6s} {'version':9s} {'n':>4s} {'mean PSNR':>10s} {'mean SSIM':>10s}")
    for s in SETS:
        for v in VERSIONS:
            sel = [r for r in rows if r["set"] == s and r["version"] == v]
            if not sel:
                continue
            print(f"{s:6s} {v:9s} {len(sel):4d} {np.mean([r['psnr'] for r in sel]):10.2f} "
                  f"{np.mean([r['ssim'] for r in sel]):10.4f}")
    print(f"-> {RESULTS/'psnr.csv'}")


if __name__ == "__main__":
    main()
