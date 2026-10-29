"""Phase 7a: Set B invention for every model, version and seed, from the audit verdicts.

Each pooled detection (> 0.25) gets a verdict about the ORIGINAL tile:
  - a detection v1 itself audited (v1 detector, sr or bicubic) keeps its own v1 final verdict;
  - every other detection takes its cluster's verdict: inherited from v1 (07_build_audit_crops.py),
    or the label given to the cluster's new crop in v2/results/audit/audit_labels.csv.

  ship      a real, unlabelled vessel: not an invention
  not_ship  invention in BOTH bands
  unsure    invention in the UPPER band only

A tile is "invented" for (model, version, threshold, band) when at least one detection above the
threshold carries an invention verdict. Rates are reported on all 150 tiles and without P1751.
A blank or invalid verdict is a hard error, as in v1. Hidden repeats give Cohen's kappa against
v1's pass-1 verdicts (same rendering) and v1's final verdicts.

REGRESSION GATE: the v1 detector's sr and bicubic rows must reproduce results/setB_audited.json and
results/setB_audited_no_P1751.json (tiles and boxes, both bands, both thresholds).

--regression-only skips the v2 models and the kappa, so the gate can run before any labelling.

Writes v2/results/setB_invention.json and v2/results/setB_invention_per_tile.csv.
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V1_RESULTS, V2_RESULTS, read_ids, v1  # noqa: E402

AUD = V2_RESULTS / "audit"
VALID = ("ship", "not_ship", "unsure")
BANDS = {"lower": {"not_ship"}, "upper": {"not_ship", "unsure"}}
CONFS = CFG["eval"]["confs"]
DOM = CFG["dominant_tile"]
V1_MODEL = "v1_offshelf"


def kappa(a, b):
    """Cohen's kappa for two raters over the same items (lists of category labels)."""
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def verdicts(regression_only):
    key = {r["crop_id"]: r for r in csv.DictReader(open(AUD / "audit_key.csv"))}
    labs = {r["crop_id"]: (r.get("verdict") or "").strip().lower()
            for r in csv.DictReader(open(AUD / "audit_labels.csv"))}
    if set(key) != set(labs):
        sys.exit("v2 audit key and labels list different crops")
    bad = [(c, v) for c, v in labs.items() if v and v not in VALID]
    if bad:
        sys.exit(f"INVALID verdicts: {bad[:10]}")
    blank = [c for c, v in labs.items() if not v]
    if blank and not regression_only:
        sys.exit(f"UNLABELLED: {len(blank)} of {len(labs)} v2 crops have no verdict ({blank[:5]}). "
                 "A blank is not a verdict. Run v2/scripts/08_label.py")
    clusters = {r["cluster"]: r for r in csv.DictReader(open(AUD / "clusters.csv"))}
    cluster_verdict = {k: (c["verdict"] or labs.get(c["crop_id"], "")) for k, c in clusters.items()}
    v1key = {(r["version"], r["tile"], int(r["box_idx"])): r["crop_id"]
             for r in csv.DictReader(open(V1_RESULTS / "audit_key.csv"))}
    v1final = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(V1_RESULTS / "audit_labels.csv"))}
    v1pass1 = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(V1_RESULTS / "audit_labels_pass1.csv"))}
    reps = [(labs[c], v1pass1[k["v1_crop_id"]], v1final[k["v1_crop_id"]])
            for c, k in key.items() if k["kind"] == "repeat" and labs[c]]
    return cluster_verdict, v1key, v1final, reps


def detection_rows(cluster_verdict, v1key, v1final, regression_only):
    rows = []
    for d in csv.DictReader(open(AUD / "detections.csv")):
        if regression_only and (d["model"] != V1_MODEL or d["version"] not in ("hr", "bicubic", "sr")):
            continue
        own = v1key.get((d["version"], d["tile"], int(d["rank"]))) if d["model"] == V1_MODEL else None
        verdict = v1final[own] if own else cluster_verdict[d["cluster"]]
        if not verdict:
            sys.exit(f"detection on {d['tile']} (cluster {d['cluster']}) has no verdict")
        rows.append({**d, "conf": float(d["conf"]), "verdict": verdict})
    return rows


def rate(hits):
    lo, hi = v1.bootstrap_pct_ci(hits, n_boot=CFG["eval"]["n_boot"], seed=CFG["eval"]["boot_seed"])
    return {"tiles": int(sum(hits)), "n_tiles": len(hits), "pct": 100.0 * sum(hits) / len(hits), "pct_ci95": [lo, hi]}


def regression_gate(res):
    bad = []
    for ref_file, subset in (("setB_audited.json", "all"), ("setB_audited_no_P1751.json", "no_dominant")):
        ref = json.loads((V1_RESULTS / ref_file).read_text())["bands"]
        for c in CONFS:
            for band in BANDS:
                for v in ("sr", "bicubic"):
                    want = ref[f"conf{c}_{band}"][v]
                    got = res[V1_MODEL][v][f"conf{c}"][band][subset]
                    if (got["tiles"], got["boxes"]) != (want["tiles_with_ship"], want["total_ships"]):
                        bad.append(f"{ref_file} conf{c} {band} {v}: v2 {got['tiles']} tiles/{got['boxes']} boxes "
                                   f"vs v1 {want['tiles_with_ship']}/{want['total_ships']}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression-only", action="store_true")
    args = ap.parse_args()
    cv, v1key, v1final, reps = verdicts(args.regression_only)
    dets = detection_rows(cv, v1key, v1final, args.regression_only)
    ids = read_ids("setB")
    subsets = {"all": ids, "no_dominant": [t for t in ids if t != DOM]}
    models = sorted({(d["model"], d["condition"], d["seed"]) for d in dets}, key=lambda m: (m[0] != V1_MODEL, m[0]))
    versions = sorted({d["version"] for d in dets}, key=lambda v: ("hr", "bicubic", "sr", "sr_greyfix").index(v))
    res, per_tile = {}, []
    for model, cond, seed in models:
        res[model] = {"condition": cond, "seed": seed}
        mine = [d for d in dets if d["model"] == model]
        for v in versions:
            for c in CONFS:
                sel = [d for d in mine if d["version"] == v and d["conf"] > c]
                for band, inv in BANDS.items():
                    boxes = Counter(d["tile"] for d in sel if d["verdict"] in inv)
                    entry = {}
                    for name, tiles in subsets.items():
                        hits = [int(boxes[t] > 0) for t in tiles]
                        entry[name] = {**rate(hits), "boxes": sum(boxes[t] for t in tiles)}
                    res[model].setdefault(v, {}).setdefault(f"conf{c}", {})[band] = entry
                    per_tile += [{"model": model, "condition": cond, "seed": seed, "version": v, "conf": c,
                                  "band": band, "tile": t, "invented_boxes": boxes[t]} for t in ids]
                res[model][v][f"conf{c}"]["verdict_boxes"] = dict(Counter(d["verdict"] for d in sel))

    bad = regression_gate(res)
    res["_regression_gate"] = {"passed": not bad, "mismatches": bad,
                               "reference": ["results/setB_audited.json", "results/setB_audited_no_P1751.json"]}
    if reps:
        new, p1, fin = zip(*reps)
        res["_kappa"] = {"n_repeats": len(reps), "vs_v1_pass1": kappa(list(new), list(p1)),
                         "vs_v1_final": kappa(list(new), list(fin)),
                         "agreement_pass1": sum(a == b for a, b in zip(new, p1)) / len(reps),
                         "pairs": [{"v2": a, "v1_pass1": b, "v1_final": c} for a, b, c in reps]}
    out = V2_RESULTS / ("setB_invention_regression.json" if args.regression_only else "setB_invention.json")
    out.write_text(json.dumps(res, indent=2))
    if not args.regression_only:
        with open(V2_RESULTS / "setB_invention_per_tile.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)

    for m in res:
        if m.startswith("_"):
            continue
        for v in versions:
            for c in CONFS:
                e = res[m][v][f"conf{c}"]
                print(f"{m:14s} {v:11s} {c:4.2f}  lower {e['lower']['all']['pct']:5.1f}% "
                      f"upper {e['upper']['all']['pct']:5.1f}%  (no P1751: {e['lower']['no_dominant']['pct']:5.1f}-"
                      f"{e['upper']['no_dominant']['pct']:5.1f}%)")
    if "_kappa" in res:
        print(f"kappa vs v1 pass 1: {res['_kappa']['vs_v1_pass1']:.3f}  vs v1 final: {res['_kappa']['vs_v1_final']:.3f}")
    if bad:
        print("\nREGRESSION GATE FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"\nregression gate passed: v1 detector invention reproduces v1's audited numbers\n-> {out}")


if __name__ == "__main__":
    main()
