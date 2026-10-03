"""Phase 3 — run the detector on every version and write the metrics.

3a  Set B (invention): ship count per tile per version per conf -> results/setB_counts.csv
3b  Set A (recovery):  model.val per version -> results/setA_metrics.json
"""
import argparse, csv, json, sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, FIGURES, DETECTOR, IMGSZ, SHIP_CLS, device  # noqa: E402

VERSIONS = ("hr", "bicubic", "sr")
CONFS = (0.25, 0.5)
CHUNK = 8


def count_ships(model, dev, paths, conf, desc):
    """-> {stem: n_ships}. Chunked: a long list source makes ultralytics build one big tensor."""
    out = {}
    with tqdm(total=len(paths), desc=desc, leave=False) as bar:
        for i in range(0, len(paths), CHUNK):
            group = paths[i:i + CHUNK]
            for p, r in zip(group, model.predict([str(x) for x in group], imgsz=IMGSZ, conf=conf,
                                                 device=dev, verbose=False)):
                out[Path(p).stem] = 0 if r.obb is None else int((r.obb.cls == SHIP_CLS).sum())
            bar.update(len(group))
    return out


def setB(model, dev):
    rows, per_version = [], {}
    for v in VERSIONS:
        imgs = sorted((DATA / "versions" / v / "setB" / "images").glob("*.png"))
        assert imgs, f"no images in versions/{v}/setB"
        for c in CONFS:
            counts = count_ships(model, dev, imgs, c, f"setB {v} conf{c}")
            per_version[(v, c)] = counts
            for tile, n in counts.items():
                rows.append({"tile": tile, "version": v, "conf": c, "n_ships": n})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "setB_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile", "version", "conf", "n_ships"])
        w.writeheader()
        w.writerows(rows)

    n_tiles = len(per_version[("hr", 0.25)])
    summary = {"n_tiles": n_tiles, "per_version": {}}
    print(f"\nSet B — {n_tiles} verified-empty tiles")
    print(f"{'version':10s} {'conf':>5s} {'tiles w/ >=1 ship':>18s} {'total invented':>15s} {'per 100 tiles':>14s}")
    for v in VERSIONS:
        for c in CONFS:
            cnt = per_version[(v, c)]
            tiles_hit = sum(1 for n in cnt.values() if n > 0)
            total = sum(cnt.values())
            per100 = 100.0 * total / n_tiles
            summary["per_version"][f"{v}_conf{c}"] = {
                "tiles_with_ship": tiles_hit, "total_ships": total,
                "tiles_with_ship_pct": 100.0 * tiles_hit / n_tiles,
                "invented_per_100_tiles": per100,
            }
            print(f"{v:10s} {c:5.2f} {tiles_hit:18d} {total:15d} {per100:14.1f}")

    # tiles where SR invented a ship -> figure candidates
    sr25 = per_version[("sr", 0.25)]
    bad = sorted((n, t) for t, n in sr25.items() if n > 0)[::-1]
    summary["sr_invented_tiles"] = [{"tile": t, "n_ships": n} for n, t in bad]
    outdir = FIGURES / "invented"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nSR invented ships on {len(bad)} tiles; saving annotated panels -> {outdir}")
    for n, t in bad:
        for v in VERSIONS:
            p = DATA / "versions" / v / "setB" / "images" / f"{t}.png"
            r = model.predict(str(p), imgsz=IMGSZ, conf=0.25, device=dev, verbose=False)[0]
            r.save(filename=str(outdir / f"{t}__{v}.png"))
    return summary


def ship_row(m):
    """Ultralytics 8.4 flattened OBBMetrics: ap_class_index/class_result sit on the metrics
    object itself, not under .box as older versions had it. Handle both."""
    src = getattr(m, "box", m)
    idx = [int(x) for x in src.ap_class_index]
    if SHIP_CLS not in idx:
        return None
    i = idx.index(SHIP_CLS)
    p, r, ap50, ap = src.class_result(i)
    return {"precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)}


def operating_conf(m):
    """Ultralytics does NOT report P/R at a fixed threshold. ap_per_class (metrics.py:92) picks
    i = smooth(f1_curve.mean(0), 0.1).argmax() -- the confidence maximising the SMOOTHED MEAN F1
    over ALL classes -- then reads every class's P and R at that same index. Recover that
    confidence so the table can say which operating point it is quoting."""
    import numpy as np
    from ultralytics.utils.metrics import smooth
    px, f1_curve = m.curves_results[1][0], np.asarray(m.curves_results[1][1])
    i = int(smooth(f1_curve.mean(0), 0.1).argmax())
    return float(np.asarray(px)[i])


def dataset_audit(v):
    """Ultralytics silently drops tiles whose label polygons fall outside [0,1]. Record how many
    tiles and instances actually entered validation so the table cannot overstate the sample."""
    import numpy as np
    c = np.load(DATA / "versions" / v / "setA" / "labels.cache", allow_pickle=True).item()
    labels = c.get("labels", [])
    msgs = [m.split(": ignoring")[0].split("/")[-1] for m in c.get("msgs", [])]
    return {
        "n_tiles_validated": len(labels),
        "n_instances_validated": int(sum(len(l["cls"]) for l in labels)),
        "n_ship_instances_validated": int(sum(int((l["cls"] == SHIP_CLS).sum()) for l in labels)),
        "tiles_dropped_by_ultralytics": msgs,
    }


def setA(model, dev):
    out = {}
    for v in VERSIONS:
        yml = DATA / "yaml" / f"{v}_setA.yaml"
        assert yml.exists(), f"missing {yml}"
        m = model.val(data=str(yml), imgsz=IMGSZ, batch=1, device=dev, plots=False,
                      project=str(RESULTS / "ultralytics"), name=f"{v}_setA", exist_ok=True,
                      verbose=False)
        row = ship_row(m)
        src = getattr(m, "box", m)
        out[v] = {"ship": row, "all_classes_mAP50": float(src.map50), "all_classes_mAP50_95": float(src.map)}
        out[v].update(dataset_audit(v))
        out[v]["operating_conf_max_mean_f1"] = operating_conf(m)
        print(f"{v:10s} ship P={row['precision']:.3f} R={row['recall']:.3f} "
              f"mAP50={row['mAP50']:.3f} mAP50-95={row['mAP50_95']:.3f}")
    with open(RESULTS / "setA_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["3a", "3b", "all"], default="all")
    args = ap.parse_args()

    from ultralytics import YOLO
    dev = device()
    model = YOLO(DETECTOR)

    if args.stage in ("3a", "all"):
        s = setB(model, dev)
        with open(RESULTS / "setB_summary.json", "w") as f:
            json.dump(s, f, indent=2)
    if args.stage in ("3b", "all"):
        setA(model, dev)


if __name__ == "__main__":
    main()
