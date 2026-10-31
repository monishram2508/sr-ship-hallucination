"""Phase 2/3: assemble the one folder that gets uploaded as a PRIVATE Kaggle dataset.

  code/scripts/common.py, code/scripts/03_degrade_sr.py   v1's code, copied unchanged, so the GPU
                                                          side degrades and super-resolves with
                                                          the exact v1 functions
  code/v2/{common_v2.py, config.yaml, kaggle_train.py, requirements.txt}
  code/weights/RealESRGAN_x4plus.pth                      SR weights (BSD-3), so no download step
  data/images/*.jpg, data/labels/*.txt                    the 900 HR tiles and ship-only labels
  data/trainset_manifest.csv                              split, kind, HR and LR hashes
  data/check_sr/*.png                                     5 tiles super-resolved on this machine
  dataset-metadata.json                                   for `kaggle datasets create` (private)

No Set A / Set B imagery is uploaded: predictions run locally. DOTA is academic-use only, so the
dataset must stay private. Prints the exact upload size and records it in v2/results/.
"""
import csv, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, ROOT, V2, V2_DATA, V2_RESULTS, repo_path  # noqa: E402

UP = V2 / "kaggle_upload"
SLUG = "sr-ship-v2-trainset"


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    rows = list(csv.DictReader(open(V2_DATA / "trainset_manifest.csv")))
    assert rows and "lr_md5" in rows[0], "run 02_degrade_train.py first (manifest has no lr_md5)"
    if UP.exists():
        shutil.rmtree(UP)
    for f in ("scripts/common.py", "scripts/03_degrade_sr.py"):
        copy(ROOT / f, UP / "code" / f)
    for f in ("common_v2.py", "config.yaml", "kaggle_train.py", "requirements.txt"):
        copy(V2 / f, UP / "code" / "v2" / f)
    copy(repo_path(CFG["v1"]["sr_weights"]), UP / "code" / "weights" / "RealESRGAN_x4plus.pth")

    ts = V2_DATA / "trainset"
    for r in rows:
        copy(ts / "images" / f"{r['tile']}.jpg", UP / "data" / "images" / f"{r['tile']}.jpg")
        copy(ts / "labels" / f"{r['tile']}.txt", UP / "data" / "labels" / f"{r['tile']}.txt")
    for p in sorted((ts / "check_sr").glob("*.png")):
        copy(p, UP / "data" / "check_sr" / p.name)
    copy(V2_DATA / "trainset_manifest.csv", UP / "data" / "trainset_manifest.csv")
    (UP / "dataset-metadata.json").write_text(json.dumps(
        {"title": "sr ship v2 trainset", "id": f"YOUR_KAGGLE_USERNAME/{SLUG}",
         "licenses": [{"name": "other"}]}, indent=2))

    parts = {k: size(UP / k) for k in ("code", "data")}
    parts["data/images"] = size(UP / "data" / "images")
    total = size(UP)
    rep = {"total_bytes": total, "total_mb": round(total / 2**20, 1),
           "parts_mb": {k: round(v / 2**20, 1) for k, v in parts.items()},
           "n_tiles": len(rows), "slug": SLUG}
    (V2_RESULTS / "kaggle_upload_size.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"-> {UP.relative_to(ROOT)}  (upload as a PRIVATE dataset; DOTA is academic-use only)")


if __name__ == "__main__":
    main()
