"""Phase 5a: Set A precision, recall, F1, TP/FP/FN at fixed thresholds, and AP50.

Matching is v1's (10_fixed_threshold.py): rotated-box IoU via ultralytics' batch_probiou, greedy,
highest confidence first, IoU >= 0.5. It runs once per tile over every detection >= 0.01; because
it is greedy in confidence order, the counts at any threshold are a prefix of that one pass, and
AP50 comes from the same pass.

Ground truth is the DOTA ship labels (class 1 in v1's 15-class label files) for every model, v1 and
v2 alike.

REGRESSION GATE: the v1 detector called exactly as v1 called it (v1_offshelf_v1settings: conf 0.25,
max_det 300) must reproduce results/setA_fixed_threshold.json exactly (TP, FP, FN at 0.25 and 0.5
on hr, bicubic, sr). If not, this script exits non-zero. The reference row v1_offshelf (conf 0.01,
max_det 1000) is then compared tile by tile with it, which isolates v1's max_det truncation.

Writes v2/results/setA_metrics.json and v2/results/setA_per_tile.csv (per tile TP/FP/GT, used by the
bootstrap in 10_stats.py).
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import (CFG, V1_RESULTS, V1_VERSIONS, V2_RESULTS, read_ids, gt_xywhr, greedy_tp,  # noqa: E402
                       ap50, above, load_predictions, prediction_files, models_in, max_det_hits,
                       print_max_det_hits)

CONFS = CFG["eval"]["confs"]
IOU_T = CFG["eval"]["iou"]


def evaluate(preds, model, version, ids, gts):
    """-> (summary dict, per-tile rows)."""
    tile_rows, all_conf, all_tp = [], [], []
    for t in ids:
        xywhr, _, conf = preds.get((model, "setA", version), {}).get(t, (np.zeros((0, 5)), None, np.zeros(0)))
        tp = greedy_tp(gts[t], xywhr, IOU_T)
        all_conf.append(conf)
        all_tp.append(tp)
        for c in CONFS:
            k = above(conf, c)
            tile_rows.append({"tile": t, "conf": c, "tp": int(tp[k].sum()),
                              "fp": int(k.sum() - tp[k].sum()), "n_gt": len(gts[t])})
    n_gt = sum(len(g) for g in gts.values())
    out = {"n_tiles": len(ids), "n_gt": n_gt,
           "ap50": ap50(np.concatenate(all_tp), np.concatenate(all_conf), n_gt)}
    for c in CONFS:
        sel = [r for r in tile_rows if r["conf"] == c]
        TP, FP = sum(r["tp"] for r in sel), sum(r["fp"] for r in sel)
        FN = n_gt - TP
        p = TP / (TP + FP) if TP + FP else 0.0
        r = TP / n_gt if n_gt else 0.0
        out[f"conf{c}"] = {"TP": TP, "FP": FP, "FN": FN, "precision": p, "recall": r,
                           "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out, tile_rows


def regression_gate(res):
    ref = json.loads((V1_RESULTS / "setA_fixed_threshold.json").read_text())["results"]
    bad = []
    for v in ("hr", "bicubic", "sr"):
        for c in CONFS:
            got, want = res["v1_offshelf_v1settings"][v][f"conf{c}"], ref[f"{v}_conf{c}"]
            for k in ("TP", "FP", "FN"):
                if got[k] != want[k]:
                    bad.append(f"{v} conf{c} {k}: v2 {got[k]} vs v1 {want[k]}")
    return bad


def truncation(per_tile):
    """Tiles where v1's call (max_det 300) kept fewer ship boxes than the uncapped reference row."""
    key = lambda r: (r["version"], r["conf"], r["tile"])  # noqa: E731
    capped = {key(r): r for r in per_tile if r["model"] == "v1_offshelf_v1settings"}
    out = {}
    for r in per_tile:
        if r["model"] != "v1_offshelf" or key(r) not in capped:
            continue
        o = capped[key(r)]
        if (r["tp"], r["fp"]) != (o["tp"], o["fp"]):
            out.setdefault(f"{r['version']}_conf{r['conf']}", []).append(
                {"tile": r["tile"], "boxes_v1_call": o["tp"] + o["fp"], "boxes_uncapped": r["tp"] + r["fp"],
                 "tp_v1_call": o["tp"], "tp_uncapped": r["tp"], "n_gt": r["n_gt"]})
    return out


def main():
    files = prediction_files()
    preds = load_predictions(files)
    ids = read_ids("setA")
    lab = V1_VERSIONS / "hr" / "setA" / "labels"
    gts = {t: gt_xywhr(lab / f"{t}.txt", CFG["v1"]["ship_cls"]) for t in ids}
    order = ("hr", "bicubic", "sr", "sr_greyfix")
    res, per_tile = {}, []
    for model, cond, seed in tqdm(models_in(files), desc="Set A"):
        res[model] = {"condition": cond, "seed": seed}
        # every model is run on hr/bicubic/sr; a version with zero boxes still gets scored (recall 0)
        mine = sorted({v for (m, s, v) in preds if m == model and s == "setA"} | {"hr", "bicubic", "sr"},
                      key=order.index)
        for v in mine:
            res[model][v], rows = evaluate(preds, model, v, ids, gts)
            per_tile += [{"model": model, "condition": cond, "seed": seed, "version": v, **r} for r in rows]
    versions = sorted({v for (_, s, v) in preds if s == "setA"}, key=order.index)

    bad = regression_gate(res)
    res["_regression_gate"] = {"passed": not bad, "mismatches": bad, "checked_model": "v1_offshelf_v1settings",
                               "reference": "results/setA_fixed_threshold.json"}
    res["_v1_max_det_truncation"] = truncation(per_tile)
    res["_max_det_hits"] = max_det_hits(files, preds, "setA", min(CONFS))
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    (V2_RESULTS / "setA_metrics.json").write_text(json.dumps(res, indent=2))
    with open(V2_RESULTS / "setA_per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "seed", "version", "tile", "conf", "tp", "fp", "n_gt"])
        w.writeheader()
        w.writerows(per_tile)

    print(f"{'model':14s} {'version':11s} {'conf':>5s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s} {'AP50':>6s}")
    for m, d in res.items():
        if m.startswith("_"):
            continue
        for v in [v for v in versions if v in d]:
            for c in CONFS:
                e = d[v][f"conf{c}"]
                print(f"{m:14s} {v:11s} {c:5.2f} {e['TP']:6d} {e['FP']:6d} {e['FN']:6d} "
                      f"{e['precision']:6.3f} {e['recall']:6.3f} {e['f1']:6.3f} {d[v]['ap50']:6.3f}")
    if bad:
        print("\nREGRESSION GATE FAILED: v2 matching does not reproduce v1's Set A numbers:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("\nregression gate passed: v1's own call reproduces results/setA_fixed_threshold.json exactly")
    for k, tiles in res["_v1_max_det_truncation"].items():
        lost = sum(t["boxes_uncapped"] - t["boxes_v1_call"] for t in tiles)
        print(f"v1 max_det=300 truncation, {k}: {len(tiles)} tiles, {lost} ship boxes cut "
              f"({', '.join(t['tile'] for t in tiles)})")
    print(f"-> {V2_RESULTS / 'setA_metrics.json'}")
    if not print_max_det_hits(res["_max_det_hits"], min(CONFS)):
        sys.exit(1)


if __name__ == "__main__":
    main()
