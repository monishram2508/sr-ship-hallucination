"""Phase 1a — tile the DOTA v1.0 *val* split into 1024x1024 crops.

Uses ultralytics.data.split_dota.split_images_and_labels (signature verified
with inspect: data_root, save_dir, split, crop_sizes, gaps).
Keeps only crops that are exactly 1024x1024, then flattens into
data/tiles/{images,labels}/.
"""
import shutil, sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, TILE  # noqa: E402

RAW = DATA / "DOTAv1"
SPLIT_OUT = DATA / "split_val"
TILES = DATA / "tiles"


def main():
    assert (RAW / "images" / "val").is_dir(), f"missing {RAW/'images'/'val'}"

    if not SPLIT_OUT.exists():
        from ultralytics.data.split_dota import split_images_and_labels
        SPLIT_OUT.mkdir(parents=True, exist_ok=True)
        split_images_and_labels(
            data_root=str(RAW),
            save_dir=str(SPLIT_OUT),
            split="val",
            crop_sizes=(TILE,),
            gaps=(200,),
        )
    else:
        print(f"{SPLIT_OUT} exists, skipping split")

    src_img = SPLIT_OUT / "images" / "val"
    src_lbl = SPLIT_OUT / "labels" / "val"
    print(f"raw crops: {len(list(src_img.glob('*')))} images, {len(list(src_lbl.glob('*.txt')))} labels")

    (TILES / "images").mkdir(parents=True, exist_ok=True)
    (TILES / "labels").mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    for p in tqdm(sorted(src_img.glob("*")), desc="filter to exactly 1024x1024"):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            dropped += 1
            continue
        if im.shape[0] != TILE or im.shape[1] != TILE:
            dropped += 1
            continue
        shutil.copyfile(p, TILES / "images" / p.name)
        lp = src_lbl / f"{p.stem}.txt"
        dst = TILES / "labels" / f"{p.stem}.txt"
        if lp.exists():
            shutil.copyfile(lp, dst)
        else:
            dst.write_text("")
        kept += 1

    print(f"kept {kept} tiles at {TILE}x{TILE}, dropped {dropped}")
    print(f"-> {TILES/'images'}")


if __name__ == "__main__":
    main()
