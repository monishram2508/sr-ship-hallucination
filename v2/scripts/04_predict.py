"""Phase 4: raw predictions for every model on every version of Set A and Set B.

Predicts at conf 0.01 with max_det 1000 and keeps only ship boxes; thresholds are applied later by
the evaluation scripts. One CSV per model in v2/data/predictions/ (gitignored: derived from DOTA),
columns = common_v2.PRED_FIELDS, so models trained later can be added without re-running the rest.

Models:
  v1_offshelf      the v1 detector (15 classes, ship = class 1) at conf 0.01, max_det 1000: the
                   reference row.
  v1_offshelf_v1settings
                   the same detector called exactly as v1 called it (conf 0.25, max_det 300, the
                   ultralytics default). v1's Set A numbers came from this call, and it truncates
                   the two densest Set A tiles at 300 boxes; the Set A regression gate in 05
                   checks the matching code against this row, and 05 reports the truncation.
  {cond}_s{seed}   every run folder under --runs holding weights/last.pt (single class, ship = 0).

Versions: hr, bicubic, sr from v1's folders, plus sr_greyfix (Phase 0 "v1b": SR converted back to
grey on grey tiles) when v2/data/v1b exists.
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_DATA, PRED_FIELDS, SETS, VERSIONS, read_ids, version_image, repo_path, v1  # noqa: E402

PRED_DIR = V2_DATA / "predictions"
V1B = V2_DATA / "v1b"
CHUNK = 8
RUN_RE = re.compile(r"^(M_[A-Z]+)_s(\d+)$")


def image_path(setname, version, tile):
    if version == "sr_greyfix":
        return V1B / setname / "images" / f"{tile}.png"
    return version_image(setname, version, tile)


def models(runs_dir):
    """-> [(name, condition, seed, weights, ship_cls, predict overrides, versions or None)]"""
    det, ship = repo_path(CFG["v1"]["detector"]), CFG["v1"]["ship_cls"]
    out = [("v1_offshelf", "v1", "", det, ship, {}, None),
           ("v1_offshelf_v1settings", "v1_maxdet300", "", det, ship, {"conf": 0.25, "max_det": 300}, VERSIONS)]
    if runs_dir and Path(runs_dir).is_dir():
        for d in sorted(Path(runs_dir).iterdir()):
            m = RUN_RE.match(d.name)
            w = d / "weights" / "last.pt"
            if m and w.exists():
                out.append((d.name, m.group(1), m.group(2), w, 0, {}, None))
    return out


def fmt(x):
    return f"{float(x):.9g}"


def predict_model(name, cond, seed, weights, ship_cls, versions, dev, overrides):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    pc = {**CFG["predict"], **overrides}
    rows, capped, n_img = [], [], 0
    for s in SETS:
        ids = read_ids(s)
        for v in versions:
            paths = [image_path(s, v, t) for t in ids]
            missing = [p for p in paths if not p.exists()]
            if missing:
                sys.exit(f"{name}: {len(missing)} missing images for {s}/{v}, e.g. {missing[0]}")
            for i in tqdm(range(0, len(paths), CHUNK), desc=f"{name} {s} {v}", leave=False):
                group = paths[i:i + CHUNK]
                res = model.predict([str(p) for p in group], imgsz=pc["imgsz"], conf=pc["conf"],
                                    max_det=pc["max_det"], device=dev, verbose=False)
                for p, r in zip(group, res):
                    n_img += 1
                    if r.obb is None or not len(r.obb):
                        continue
                    if len(r.obb) >= pc["max_det"]:
                        capped.append(f"{s}/{v}/{p.stem}")
                    keep = (r.obb.cls == ship_cls).cpu().numpy()
                    conf = r.obb.conf.cpu().numpy()[keep]
                    poly = r.obb.xyxyxyxy.cpu().numpy()[keep].reshape(-1, 8)
                    xywhr = r.obb.xywhr.cpu().numpy()[keep]
                    for c, pl, xr in zip(conf, poly, xywhr):
                        rows.append([name, cond, seed, s, v, p.stem, fmt(c),
                                     *[fmt(z) for z in pl], *[fmt(z) for z in xr]])
    return rows, capped, n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(V2_DATA.parent / "runs"),
                    help="folder of unzipped training runs, one {cond}_s{seed} folder each")
    ap.add_argument("--only", default="", help="comma-separated model names to (re)predict")
    ap.add_argument("--force", action="store_true", help="overwrite existing prediction files")
    args = ap.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    versions = list(VERSIONS) + (["sr_greyfix"] if (V1B / "setB" / "images").is_dir() else [])
    dev = v1.device()
    only = {x for x in args.only.split(",") if x}
    for name, cond, seed, w, cls, overrides, own_versions in models(args.runs):
        if only and name not in only:
            continue
        out = PRED_DIR / f"{name}.csv"
        if out.exists() and not args.force:
            print(f"{name}: {out.name} exists, skipping (--force to redo)")
            continue
        t0 = time.perf_counter()
        rows, capped, n_img = predict_model(name, cond, seed, w, cls, own_versions or versions, dev, overrides)
        tmp = out.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(PRED_FIELDS)
            wr.writerows(rows)
        tmp.replace(out)
        print(f"{name}: {len(rows)} ship boxes on {n_img} images in {time.perf_counter() - t0:.0f}s "
              f"-> {out.relative_to(V2_DATA.parent.parent)}")
        if capped:
            print(f"  !! {len(capped)} images hit max_det: {capped[:5]}")
        (PRED_DIR / f"{name}.capped.json").write_text(json.dumps(capped))


if __name__ == "__main__":
    main()
