"""Phase 2, local half: degrade every training tile with v1's own degrade(), record a hash of each LR
array, and super-resolve 5 check tiles here so the GPU run can be compared against this machine.

The GPU side (v2/kaggle_train.py) re-degrades the uploaded HR tiles, checks every LR hash against
this manifest (does OpenCV's blur + area resize give identical bytes on another platform?), then
makes the bicubic and Real-ESRGAN versions there. Uploading three 1024 px PNG versions would cost
~5 GB; uploading the HR JPEG tiles costs ~0.2 GB.

Adds lr_md5 to v2/data/trainset_manifest.csv (and its copy in v2/results/), writes LR PNGs to
v2/data/trainset/lr/ and the check SR tiles to v2/data/trainset/check_sr/.
"""
import csv, json, random, shutil, sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_DATA, V2_RESULTS, array_md5, load_v1_script, v1  # noqa: E402

TS_DIR = V2_DATA / "trainset"
MANIFEST = V2_DATA / "trainset_manifest.csv"
N_CHECK = 5


def main():
    m03 = load_v1_script("03_degrade_sr")
    rows = list(csv.DictReader(open(MANIFEST)))
    (TS_DIR / "lr").mkdir(parents=True, exist_ok=True)
    for r in tqdm(rows, desc="degrade"):
        hr = cv2.imread(str(TS_DIR / "images" / f"{r['tile']}.jpg"), cv2.IMREAD_COLOR)
        assert array_md5(hr) == r["hr_md5"], f"{r['tile']}: HR changed since 01_build_trainset.py"
        lr = m03.degrade(hr)
        r["lr_md5"] = array_md5(lr)
        cv2.imwrite(str(TS_DIR / "lr" / f"{r['tile']}.png"), lr)

    fields = list(rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    shutil.copyfile(MANIFEST, V2_RESULTS / "trainset_manifest.csv")

    check = sorted(random.Random(CFG["seed"]).sample(
        sorted(r["tile"] for r in rows if r["kind"] == "pos" and r["split"] == "train"), N_CHECK))
    out = TS_DIR / "check_sr"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dev = v1.device()
    model = m03.load_sr(dev)
    for t in check:
        lr = cv2.imread(str(TS_DIR / "lr" / f"{t}.png"), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out / f"{t}.png"), m03.run_sr(model, lr, dev))
    (V2_RESULTS / "trainset_check_tiles.json").write_text(json.dumps({"device": dev, "tiles": check}, indent=2))
    print(f"{len(rows)} LR tiles hashed; {N_CHECK} check tiles super-resolved on {dev}: {check}")


if __name__ == "__main__":
    main()
