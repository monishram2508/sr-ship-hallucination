"""Phase 5b: Set B firing, for every model and version, including the original images.

  raw firing        % of tiles with >= 1 ship detection above the threshold.
  beyond original   % of tiles where the model fired on this version but NOT on the original (hr),
                    same model, same threshold. Set B was verified empty with the v1 detector only,
                    so a v2 model can fire on hr; this is the number that still means "fired on
                    something the original did not show it".

REGRESSION GATE: the v1 detector's per-tile ship counts at 0.25 and 0.5 on hr, bicubic and sr must
equal results/setB_counts.csv exactly. If not, this script exits non-zero.

Writes v2/results/setB_firing.json and v2/results/setB_per_tile_counts.csv.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import (CFG, V1_RESULTS, V2_RESULTS, read_ids, above, load_predictions,  # noqa: E402
                       prediction_files, models_in, max_det_hits, print_max_det_hits, v1)

CONFS = CFG["eval"]["confs"]


def counts(preds, model, version, ids, c):
    tiles = preds.get((model, "setB", version), {})
    return {t: int(above(tiles[t][2], c).sum()) if t in tiles else 0 for t in ids}


def rate(hits):
    lo, hi = v1.bootstrap_pct_ci(hits, n_boot=CFG["eval"]["n_boot"], seed=CFG["eval"]["boot_seed"])
    return {"tiles": int(sum(hits)), "n_tiles": len(hits), "pct": 100.0 * sum(hits) / len(hits),
            "pct_ci95": [lo, hi]}


def regression_gate(per):
    want = {(r["tile"], r["version"], float(r["conf"])): int(r["n_ships"])
            for r in csv.DictReader(open(V1_RESULTS / "setB_counts.csv"))}
    bad = [f"{t} {v} conf{c}: v2 {per[('v1_offshelf', v, c)][t]} vs v1 {n}"
           for (t, v, c), n in want.items() if per[("v1_offshelf", v, c)][t] != n]
    return bad, len(want)


def main():
    files = prediction_files()
    preds = load_predictions(files)
    ids = read_ids("setB")
    versions = sorted({v for (_, s, v) in preds if s == "setB"} | {"hr", "bicubic", "sr"},
                      key=lambda v: ("hr", "bicubic", "sr", "sr_greyfix").index(v))
    per, res, rows = {}, {}, []
    all_models = models_in(files)
    # a model is only scored on versions it was run on (v1_offshelf_v1settings has no sr_greyfix);
    # a version with no detections at all still counts, since the model may simply never have fired
    ran = {m: {v for (mm, s, v) in preds if mm == m and s == "setB"} | {"hr", "bicubic", "sr"} for m, _, _ in all_models}
    ran["v1_offshelf_v1settings"] = {"hr", "bicubic", "sr"}
    for model, cond, seed in all_models:
        res[model] = {"condition": cond, "seed": seed}
        mine = [v for v in versions if v in ran[model]]
        for c in CONFS:
            for v in mine:
                per[(model, v, c)] = counts(preds, model, v, ids, c)
            hr_fired = np.array([per[(model, "hr", c)][t] > 0 for t in ids], int)
            for v in mine:
                cnt = per[(model, v, c)]
                fired = np.array([cnt[t] > 0 for t in ids], int)
                e = {"raw": rate(fired), "boxes": int(sum(cnt.values()))}
                if v != "hr":
                    e["beyond_original"] = rate(fired * (1 - hr_fired))
                res[model].setdefault(v, {})[f"conf{c}"] = e
                rows += [{"model": model, "condition": cond, "seed": seed, "version": v, "conf": c,
                          "tile": t, "n_ships": cnt[t]} for t in ids]

    bad, n_checked = regression_gate(per)
    res["_regression_gate"] = {"passed": not bad, "n_checked": n_checked, "mismatches": bad[:50],
                               "reference": "results/setB_counts.csv"}
    res["_max_det_hits"] = max_det_hits(files, preds, "setB", min(CONFS))
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    (V2_RESULTS / "setB_firing.json").write_text(json.dumps(res, indent=2))
    with open(V2_RESULTS / "setB_per_tile_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "seed", "version", "conf", "tile", "n_ships"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':14s} {'version':11s} {'conf':>5s} {'fired':>6s} {'%':>6s} {'95% CI':>14s} "
          f"{'beyond hr %':>11s} {'boxes':>6s}")
    for m, d in res.items():
        if m.startswith("_"):
            continue
        for c in CONFS:
            for v in [v for v in versions if v in d]:
                e = d[v][f"conf{c}"]
                bo = f"{e['beyond_original']['pct']:10.1f}%" if "beyond_original" in e else f"{'-':>11s}"
                print(f"{m:14s} {v:11s} {c:5.2f} {e['raw']['tiles']:6d} {e['raw']['pct']:5.1f}% "
                      f"[{e['raw']['pct_ci95'][0]:5.1f},{e['raw']['pct_ci95'][1]:5.1f}] {bo} {e['boxes']:6d}")
    if bad:
        print(f"\nREGRESSION GATE FAILED ({len(bad)} of {n_checked} per-tile counts differ from v1):")
        for b in bad[:20]:
            print("  " + b)
        sys.exit(1)
    print(f"\nregression gate passed: all {n_checked} v1 per-tile counts reproduced exactly")
    print(f"-> {V2_RESULTS / 'setB_firing.json'}")
    if not print_max_det_hits(res["_max_det_hits"], min(CONFS)):
        sys.exit(1)


if __name__ == "__main__":
    main()
