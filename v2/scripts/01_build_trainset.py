"""Phase 1: a small, ship-only training set from DOTA-v1.0 TRAIN.

Set A and Set B come from the val split, so nothing from val is read here.

Tiling is v1's code path: ultralytics' get_windows / get_window_obj (1024 px, gap 200, IoF 0.7) and
crop_and_save, which writes each crop as a JPEG exactly as v1's 01_tile.py got them. Only the
selected windows are cropped, and only windows lying fully inside the image (exactly 1024x1024,
the same filter v1 applied).

  positive        >= 1 ship assigned to the window (IoF >= 0.7)
  negative        NO ship polygon touching the window at all (IoF > 0). Stricter than "no ship
                  assigned": with IoF 0.7 a window can show a cut-off ship and carry no ship
                  label, which would teach the model that a ship is background.
  hard negative   negative with >= 1 harbor, large-vehicle or small-vehicle assigned
  random negative drawn from the negatives not already chosen as hard

Sampled with seed 0: 600 positives, 150 hard negatives, 150 random negatives. Split 90/10 by source
image (seeded shuffle of images, greedy until the val side holds >= 10% of tiles), so overlapping
tiles never straddle the split.

Labels: ship -> class 0, every other class dropped, coordinates clipped to [0, 1]. Ultralytics
rejects a whole image when any label point lies outside [-0.01, 1.01], and IoF-0.7 assignment
leaves edge ships partly outside the window, so unclipped labels would silently delete positives.

Writes v2/data/trainset/{images,labels_dota,labels}/, v2/data/trainset_manifest.csv (copied to
v2/results/ for the record) and v2/results/trainset_summary.json.
"""
import csv, json, random, shutil, sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_DATA, V2_RESULTS, TILE, repo_path, array_md5  # noqa: E402

TS = CFG["trainset"]
SHIP = TS["ship_cls_dota"]
OUT = V2_DATA / "trainset"
MANIFEST = V2_DATA / "trainset_manifest.csv"
FIELDS = ["tile", "source", "x0", "y0", "split", "kind", "has_hard_class", "n_ships", "n_clipped", "hr_md5"]


def enumerate_windows(annos):
    """-> list of window records over every train image, plus {source: (anno, windows, objs)}."""
    from ultralytics.data.split_dota import get_windows, get_window_obj, bbox_iof
    recs, per_image = [], {}
    for anno in tqdm(annos, desc="windows"):
        h, w = anno["ori_size"]
        wins = get_windows((h, w), crop_sizes=(TILE,), gaps=(CFG["gap"],))
        objs = get_window_obj(anno, wins, iof_thr=0.7)          # denormalises anno["label"] in place
        lab = anno["label"]
        ships = lab[lab[:, 0] == SHIP] if lab.ndim == 2 and len(lab) else np.zeros((0, 9), np.float32)
        touch = (bbox_iof(ships[:, 1:], wins) > 0).any(0) if len(ships) else np.zeros(len(wins), bool)
        src = Path(anno["filepath"]).stem
        per_image[src] = (anno, wins, objs)
        for i, (x0, y0, x1, y1) in enumerate(wins.tolist()):
            if x1 > w or y1 > h:                                 # not exactly 1024x1024 once cropped
                continue
            cls = objs[i][:, 0].astype(int) if len(objs[i]) else np.zeros(0, int)
            recs.append({"tile": f"{src}__{TILE}__{x0}___{y0}", "source": src, "x0": x0, "y0": y0,
                         "win_idx": i, "n_ships": int((cls == SHIP).sum()),
                         "touches_ship": bool(touch[i]),
                         "has_hard_class": bool(np.isin(cls, TS["hard_neg_classes"]).any())})
    return recs, per_image


def sample(recs):
    rng = random.Random(CFG["seed"])
    pos = sorted(r["tile"] for r in recs if r["n_ships"] > 0)
    neg = sorted(r["tile"] for r in recs if r["n_ships"] == 0 and not r["touches_ship"])
    by = {r["tile"]: r for r in recs}
    hard_pool = [t for t in neg if by[t]["has_hard_class"]]
    n_hard = TS["n_neg"] // 2
    chosen_pos = rng.sample(pos, TS["n_pos"])
    chosen_hard = rng.sample(hard_pool, n_hard)
    rest = sorted(set(neg) - set(chosen_hard))
    chosen_rand = rng.sample(rest, TS["n_neg"] - n_hard)
    kind = {**{t: "pos" for t in chosen_pos}, **{t: "hard_neg" for t in chosen_hard},
            **{t: "rand_neg" for t in chosen_rand}}
    pools = {"windows_exact_1024": len(recs), "positive_pool": len(pos), "negative_pool_strict": len(neg),
             "negative_excluded_touching_ship": sum(1 for r in recs if r["n_ships"] == 0 and r["touches_ship"]),
             "hard_negative_pool": len(hard_pool)}
    return kind, pools


def split_by_source(kind, by):
    rng = random.Random(CFG["seed"])
    sources = sorted({by[t]["source"] for t in kind})
    rng.shuffle(sources)
    target = int(np.ceil(TS["val_frac"] * len(kind)))
    n_by_src = {s: sum(1 for t in kind if by[t]["source"] == s) for s in sources}
    val, n = set(), 0
    for s in sources:
        if n >= target:
            break
        val.add(s)
        n += n_by_src[s]
    return {t: ("val" if by[t]["source"] in val else "train") for t in kind}


def ship_only_label(dota_lbl):
    """15-class normalised label file -> (ship-only lines with class 0, clipped to [0,1], n_clipped)."""
    lines, clipped = [], 0
    if dota_lbl.exists():
        for line in dota_lbl.read_text().splitlines():
            p = line.split()
            if len(p) != 9 or int(p[0]) != SHIP:
                continue
            xy = np.asarray(p[1:], np.float64)
            c = np.clip(xy, 0.0, 1.0)
            clipped += int(not np.array_equal(c, xy))
            lines.append("0 " + " ".join(f"{v:.6g}" for v in c))
    return lines, clipped


def rejected_without_clipping(rows):
    """Tiles whose UNclipped ship labels would make ultralytics drop the whole image (any point
    outside [-0.01, 1.01]), and the ships on them: what clipping saves."""
    bad = []
    for r in rows:
        f = OUT / "labels_dota" / f"{r['tile']}.txt"
        pts = [np.asarray(l.split()[1:], float) for l in (f.read_text().splitlines() if f.exists() else [])
               if l.split() and int(l.split()[0]) == SHIP]
        if pts and (max(p.max() for p in pts) > 1.01 or min(p.min() for p in pts) < -0.01):
            bad.append(r)
    return {"tiles": len(bad), "ships_on_them": sum(r["n_ships"] for r in bad)}


def main():
    from ultralytics.data.split_dota import load_yolo_dota, crop_and_save
    root = repo_path(CFG["dota_root"])
    assert (root / "images" / "train").is_dir(), f"DOTA train split missing under {root}"
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("images", "labels_dota", "labels"):
        (OUT / d).mkdir(parents=True)

    annos = sorted(load_yolo_dota(str(root), split="train"), key=lambda a: a["filepath"])
    recs, per_image = enumerate_windows(annos)
    by = {r["tile"]: r for r in recs}
    kind, pools = sample(recs)
    split = split_by_source(kind, by)

    for src in tqdm(sorted({by[t]["source"] for t in kind}), desc="crop"):
        anno, wins, objs = per_image[src]
        idx = [by[t]["win_idx"] for t in kind if by[t]["source"] == src]
        crop_and_save(anno, wins[idx], [objs[i] for i in idx], str(OUT / "images"), str(OUT / "labels_dota"))

    rows = []
    for t in tqdm(sorted(kind), desc="labels"):
        r = by[t]
        lines, n_clip = ship_only_label(OUT / "labels_dota" / f"{t}.txt")
        assert len(lines) == r["n_ships"], f"{t}: {len(lines)} ship lines vs {r['n_ships']} assigned"
        (OUT / "labels" / f"{t}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        im = cv2.imread(str(OUT / "images" / f"{t}.jpg"), cv2.IMREAD_COLOR)
        assert im is not None and im.shape[:2] == (TILE, TILE), f"{t}: bad crop {None if im is None else im.shape}"
        rows.append({"tile": t, "source": r["source"], "x0": r["x0"], "y0": r["y0"], "split": split[t],
                     "kind": kind[t], "has_hard_class": int(r["has_hard_class"]), "n_ships": r["n_ships"],
                     "n_clipped": n_clip, "hr_md5": array_md5(im)})

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANIFEST, V2_RESULTS / "trainset_manifest.csv")

    summ = {"pools": pools, "n_tiles": len(rows), "n_sources": len({r["source"] for r in rows}),
            "by_split_kind": {}, "ships": {}, "clipped_polygons": sum(r["n_clipped"] for r in rows),
            "positives_ultralytics_would_drop_unclipped": rejected_without_clipping(rows),
            "val_sources": sorted({r["source"] for r in rows if r["split"] == "val"})}
    for sp in ("train", "val"):
        for k in ("pos", "hard_neg", "rand_neg"):
            summ["by_split_kind"][f"{sp}_{k}"] = sum(1 for r in rows if r["split"] == sp and r["kind"] == k)
        summ["ships"][sp] = sum(r["n_ships"] for r in rows if r["split"] == sp)
    (V2_RESULTS / "trainset_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "val_sources"}, indent=2))
    print(f"-> {MANIFEST.relative_to(V2_DATA.parent.parent)}")


if __name__ == "__main__":
    main()
