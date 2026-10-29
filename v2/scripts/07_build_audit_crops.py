"""Phase 6: pool every Set B detection, cluster it, inherit v1's verdicts, and build blind crops for
the regions v1 never audited.

  1. Recover v1's audited polygons. v1's audit key stores crop windows, not boxes, so the v1
     detector's Phase 4 predictions are re-ranked exactly as v1's 08_audit_boxes.py ranked them
     (ship boxes > 0.25, confidence-descending) and every key row is checked: same confidence to 4
     decimals, and v1's own crop_box() of the recovered polygon gives the key's crop window. Any
     mismatch is a hard error.
  2. Pool every ship detection > 0.25 on Set B across all models, seeds and versions (original
     included). Cluster per tile, highest confidence first: a detection joins the cluster whose
     seed box it overlaps most if probIoU >= 0.5, else it seeds a new cluster.
  3. A cluster inherits a v1 verdict when its seed has probIoU >= 0.5 with a v1-audited box (best
     match wins), or else when it contains a v1-audited detection (highest-confidence one wins).
     Verdicts describe the ORIGINAL tile, so they transfer across versions and models.
  4. Every other cluster becomes a new crop of the original tile around its seed box, drawn with
     v1's own render(). 25 v1 crops, stratified by pass-1 verdict, are re-rendered the same way and
     mixed in as hidden repeats (for Cohen's kappa). All crops are shuffled with seed 0 BEFORE ids
     are assigned, so an id carries no information.

Writes v2/results/audit/{clusters.csv, detections.csv, audit_key.csv, audit_labels.csv,
build_summary.json} and crops to v2/results/audit/crops/ (gitignored). Refuses to overwrite
audit_labels.csv once it holds verdicts.
"""
import argparse, csv, json, random, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import (CFG, V1_RESULTS, V2_RESULTS, read_ids, version_image, probiou, above,  # noqa: E402
                       load_predictions, prediction_files, models_in, load_v1_script)

A = CFG["audit"]
AUD = V2_RESULTS / "audit"
CROPS = AUD / "crops"
LABELS = AUD / "audit_labels.csv"
V1_MODEL = "v1_offshelf"


def v1_audited(preds, m08):
    """-> {tile: [(xywhr, poly, v1_crop_id, final_verdict, pass1_verdict)]}, checked against the key."""
    key = list(csv.DictReader(open(V1_RESULTS / "audit_key.csv")))
    final = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(V1_RESULTS / "audit_labels.csv"))}
    pass1 = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(V1_RESULTS / "audit_labels_pass1.csv"))}
    out, bad = {}, []
    for r in key:
        xywhr, poly, conf = preds[(V1_MODEL, "setB", r["version"])][r["tile"]]
        k = above(conf, 0.25)
        i = int(r["box_idx"])
        if i >= k.sum():
            bad.append(f"{r['crop_id']}: box_idx {i} but only {k.sum()} boxes")
            continue
        win = m08.crop_box({"poly": poly[i]})
        want = tuple(int(r[f"crop_{c}"]) for c in ("x0", "y0", "x1", "y1"))
        if round(float(conf[i]), 4) != float(r["conf"]) or tuple(win) != want:
            bad.append(f"{r['crop_id']}: conf {conf[i]:.4f} vs {r['conf']}, window {win} vs {want}")
        out.setdefault(r["tile"], []).append((xywhr[i], poly[i], r["crop_id"], final[r["crop_id"]],
                                              pass1[r["crop_id"]], r["version"], i))
    if bad:
        sys.exit("v1 audit key does not match the recovered v1 boxes:\n  " + "\n  ".join(bad[:20]))
    return out, key


def pool(preds, models, versions, ids):
    dets = []
    for model, cond, seed in models:
        for v in versions:
            tiles = preds.get((model, "setB", v), {})
            for t in ids:
                if t not in tiles:
                    continue
                xywhr, poly, conf = tiles[t]
                for j in np.flatnonzero(above(conf, A["conf"])):
                    dets.append({"model": model, "condition": cond, "seed": seed, "version": v, "tile": t,
                                 "rank": int(j), "conf": float(conf[j]), "xywhr": xywhr[j], "poly": poly[j]})
    return dets


def cluster(dets):
    """Greedy, per tile, highest confidence first. Returns clusters; sets d['cluster'] on each det."""
    clusters = []
    by_tile = {}
    for d in dets:
        by_tile.setdefault(d["tile"], []).append(d)
    for t in sorted(by_tile):
        seeds = []
        for d in sorted(by_tile[t], key=lambda d: -d["conf"]):
            if seeds:
                iou = probiou(d["xywhr"][None], np.stack([c["seed"]["xywhr"] for c in seeds]))[0]
                k = int(iou.argmax())
                if iou[k] >= A["cluster_iou"]:
                    d["cluster"] = seeds[k]["id"]
                    seeds[k]["members"].append(d)
                    continue
            c = {"id": f"k{len(clusters) + 1:04d}", "tile": t, "seed": d, "members": [d]}
            d["cluster"] = c["id"]
            seeds.append(c)
            clusters.append(c)
    return clusters


def inherit(clusters, v1box):
    """Sets c['verdict'] and c['source'] ('v1_match' | 'v1_member' | 'new')."""
    v1_ids = {(b[5], t, b[6]): b for t, bs in v1box.items() for b in bs}
    for c in clusters:
        c["verdict"], c["source"], c["v1_crop_id"] = "", "new", ""
        boxes = v1box.get(c["tile"], [])
        if boxes:
            iou = probiou(c["seed"]["xywhr"][None], np.stack([b[0] for b in boxes]))[0]
            k = int(iou.argmax())
            if iou[k] >= A["inherit_iou"]:
                c["verdict"], c["source"], c["v1_crop_id"] = boxes[k][3], "v1_match", boxes[k][2]
                continue
        for d in sorted(c["members"], key=lambda d: -d["conf"]):
            b = v1_ids.get((d["version"], d["tile"], d["rank"])) if d["model"] == V1_MODEL else None
            if b is not None:
                c["verdict"], c["source"], c["v1_crop_id"] = b[3], "v1_member", b[2]
                break


def repeats(key, v1box):
    """25 v1 crops stratified by pass-1 verdict (largest-remainder allocation), seed 0."""
    rng = random.Random(CFG["seed"])
    box_of = {b[2]: (t, b) for t, bs in v1box.items() for b in bs}
    groups = {}
    for r in key:
        groups.setdefault(box_of[r["crop_id"]][1][4], []).append(r["crop_id"])
    n, total = A["n_repeats"], len(key)
    quota = {g: n * len(ids) / total for g, ids in groups.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: -(quota[g] - alloc[g]))[: n - sum(alloc.values())]:
        alloc[g] += 1
    picked = []
    for g in sorted(groups):
        picked += rng.sample(sorted(groups[g]), alloc[g])
    return [(c, *box_of[c]) for c in sorted(picked)], alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite audit_labels.csv even if it holds verdicts")
    args = ap.parse_args()
    if LABELS.exists() and not args.force:
        done = [r for r in csv.DictReader(open(LABELS)) if (r.get("verdict") or "").strip()]
        if done:
            sys.exit(f"REFUSING: {LABELS} already holds {len(done)} verdicts. Pass --force to rebuild.")

    m08 = load_v1_script("08_audit_boxes")
    files = prediction_files()
    preds = load_predictions(files)
    ids = read_ids("setB")
    versions = sorted({v for (_, s, v) in preds if s == "setB"},
                      key=lambda v: ("hr", "bicubic", "sr", "sr_greyfix").index(v))
    v1box, key = v1_audited(preds, m08)
    # v1_offshelf_v1settings duplicates v1_offshelf's Set B boxes exactly (06's gate), so it is not pooled
    pooled = [m for m in models_in(files) if m[1] != "v1_maxdet300"]
    dets = pool(preds, pooled, versions, ids)
    clusters = cluster(dets)
    inherit(clusters, v1box)
    new = [c for c in clusters if c["source"] == "new"]
    reps, alloc = repeats(key, v1box)

    crops = [{"kind": "new", "tile": c["tile"], "cluster": c["id"], "v1_crop_id": "", "poly": c["seed"]["poly"]}
             for c in new]
    crops += [{"kind": "repeat", "tile": t, "cluster": "", "v1_crop_id": cid, "poly": b[1]} for cid, t, b in reps]
    random.Random(CFG["seed"]).shuffle(crops)                 # blind: shuffle BEFORE numbering
    for i, c in enumerate(crops, 1):
        c["crop_id"] = f"v2c{i:03d}"
    crop_of_cluster = {c["cluster"]: c["crop_id"] for c in crops if c["kind"] == "new"}

    CROPS.mkdir(parents=True, exist_ok=True)
    for old in CROPS.glob("*.png"):
        old.unlink()
    hr_cache = {}
    for c in tqdm(crops, desc="crops"):
        if c["tile"] not in hr_cache:
            hr_cache = {c["tile"]: cv2.imread(str(version_image("setB", "hr", c["tile"])), cv2.IMREAD_COLOR)}
        cv2.imwrite(str(CROPS / f"{c['crop_id']}.png"), m08.render(hr_cache[c["tile"]], c, m08.BIG, 2))

    with open(AUD / "audit_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "kind", "tile", "cluster", "v1_crop_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1"])
        for c in sorted(crops, key=lambda c: c["crop_id"]):
            w.writerow([c["crop_id"], c["kind"], c["tile"], c["cluster"], c["v1_crop_id"], *m08.crop_box(c)])
    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "verdict"])
        for c in sorted(crops, key=lambda c: c["crop_id"]):
            w.writerow([c["crop_id"], ""])
    with open(AUD / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "tile", "source", "verdict", "v1_crop_id", "crop_id", "n_members",
                    "seed_model", "seed_version", "seed_conf", *[f"seed_{k}" for k in ("cx", "cy", "w", "h", "r")]])
        for c in clusters:
            s = c["seed"]
            w.writerow([c["id"], c["tile"], c["source"], c["verdict"], c["v1_crop_id"],
                        crop_of_cluster.get(c["id"], ""), len(c["members"]), s["model"], s["version"],
                        f"{s['conf']:.6f}", *[f"{x:.4f}" for x in s["xywhr"]]])
    with open(AUD / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "seed", "version", "tile", "rank", "conf", "cluster"])
        for d in dets:
            w.writerow([d["model"], d["condition"], d["seed"], d["version"], d["tile"], d["rank"],
                        f"{d['conf']:.9g}", d["cluster"]])

    summ = {"n_detections_pooled": len(dets), "n_clusters": len(clusters),
            "clusters_by_source": dict(Counter(c["source"] for c in clusters)),
            "n_new_crops": len(new), "n_repeat_crops": len(reps), "repeat_allocation_by_pass1": alloc,
            "n_crops_to_label": len(crops), "models": [m for m, _, _ in pooled], "versions": versions}
    (AUD / "build_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"\n{len(crops)} crops to label ({len(new)} new, {len(reps)} hidden repeats). "
          "Next: .venv/bin/python v2/scripts/08_label.py")


if __name__ == "__main__":
    main()
