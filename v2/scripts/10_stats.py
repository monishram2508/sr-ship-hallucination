"""Phase 7b: the pre-registered comparisons, per seed and across seeds.

  C1 (the reviewer's)       on SR images: M_SR vs M_HR
  C2 (fair pipelines)       M_SR on SR vs M_BIC on bicubic       <- headline
  C3 (references)           M_HR on the original (ceiling), and the v1 detector's rows

Set B (tile-level yes/no): raw firing, firing beyond the original, invention (lower and upper band),
at 0.25 and 0.5, on all tiles and without P1751. Per seed: both rates, the difference, exact
McNemar on the paired tile outcomes, and a paired tile-bootstrap 95% CI of the difference.
Set A: recall, precision and F1 at 0.25 and 0.5 (paired tile-bootstrap CI of the difference, as a
ratio of sums) and AP50 (per seed only).
Across seeds: mean and SD of each rate and difference, whether the difference has the same sign in
every seed, and a tile bootstrap of the seed-averaged difference (same resampled tiles for every
seed, so the interval carries tile variability and the mean carries seed variability).

Primary endpoints (fixed in PLAN.md before any v2 result): Set B invention at conf 0.25, all tiles,
both bands; Set A recall at conf 0.25. Everything else is secondary.

Added after first results (stats.json "added_after_first_results"; none of it is a pre-registered
endpoint, and the blocks above are computed exactly as before):
  - excess over own originals: invention of M_SR on SR and M_BIC on bicubic minus the same detector
    on the original tiles, per seed, and the difference of the two excesses (v1 rows for reference);
  - seed-pairing-free checks for C1 and C2: all 9 (A seed, B seed) pairings, and a majority vote
    (a tile counts as invented when at least 2 of the 3 seeds invent on it).

Reads v2/results/{setA_per_tile.csv, setA_metrics.json, setB_per_tile_counts.csv,
setB_invention_per_tile.csv}; writes v2/results/stats.json.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_RESULTS, read_ids, v1  # noqa: E402

CONFS = CFG["eval"]["confs"]
DOM = CFG["dominant_tile"]
NB, BS = CFG["eval"]["n_boot"], CFG["eval"]["boot_seed"]
COMPARISONS = {
    "C1_sr_images_MSR_vs_MHR": (("M_SR", "sr"), ("M_HR", "sr")),
    "C2_pipelines_MSR_on_sr_vs_MBIC_on_bicubic": (("M_SR", "sr"), ("M_BIC", "bicubic")),
}
REFERENCES = [("M_HR", "hr"), ("M_SR", "hr"), ("M_BIC", "hr"), ("M_HR", "bicubic"), ("M_BIC", "sr")]
PRIMARY = ["setB_invention_lower_conf0.25_all", "setB_invention_upper_conf0.25_all", "setA_recall_conf0.25"]
EXCESS = {  # added after first results: restored-image invention minus the same detector on the originals
    "M_SR_on_sr_over_own_originals": (("M_SR", "sr"), ("M_SR", "hr")),
    "M_BIC_on_bicubic_over_own_originals": (("M_BIC", "bicubic"), ("M_BIC", "hr")),
    "M_HR_on_sr_over_own_originals": (("M_HR", "sr"), ("M_HR", "hr")),
    "v1_on_sr_over_own_originals": (("v1", "sr"), ("v1", "hr")),
    "v1_on_bicubic_over_own_originals": (("v1", "bicubic"), ("v1", "hr")),
}


def boot_idx(n):
    return np.random.default_rng(BS).integers(0, n, size=(NB, n))


def ci(stat):
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return [float(lo), float(hi)]


def setB_arrays(seeds):
    """-> {metric_key: {(condition, version, seed): 0/1 array over tiles}} for Set B."""
    ids = read_ids("setB")
    subsets = {"all": ids, "no_dominant": [t for t in ids if t != DOM]}
    fire = pd.read_csv(V2_RESULTS / "setB_per_tile_counts.csv", dtype={"seed": str}, keep_default_na=False)
    inv_path = V2_RESULTS / "setB_invention_per_tile.csv"
    inv = pd.read_csv(inv_path, dtype={"seed": str}, keep_default_na=False) if inv_path.exists() else None
    out = {}

    def put(key, df, value_col, cond, ver, seed, tiles):
        s = df.set_index("tile")[value_col].reindex(tiles).fillna(0).to_numpy()
        out.setdefault(key, {})[(cond, ver, seed)] = (s > 0).astype(int)

    for c in CONFS:
        f = fire[np.isclose(fire["conf"], c)]
        for (cond, seed, ver), g in f.groupby(["condition", "seed", "version"]):
            hr = f[(f["condition"] == cond) & (f["seed"] == seed) & (f["version"] == "hr")]
            for sub, tiles in subsets.items():
                put(f"setB_firing_raw_conf{c}_{sub}", g, "n_ships", cond, ver, seed, tiles)
                fired = g.set_index("tile")["n_ships"].reindex(tiles).fillna(0).to_numpy() > 0
                hr_f = hr.set_index("tile")["n_ships"].reindex(tiles).fillna(0).to_numpy() > 0
                out.setdefault(f"setB_firing_beyond_conf{c}_{sub}", {})[(cond, ver, seed)] = (fired & ~hr_f).astype(int)
        if inv is None:
            continue
        iv = inv[np.isclose(inv["conf"], c)]
        for (cond, seed, ver, band), g in iv.groupby(["condition", "seed", "version", "band"]):
            for sub, tiles in subsets.items():
                put(f"setB_invention_{band}_conf{c}_{sub}", g, "invented_boxes", cond, ver, seed, tiles)
    return out


def binary_comparison(arrs, a, b, seeds):
    per_seed, diffs, idx = {}, [], None
    for s in seeds:
        x, y = arrs.get((a[0], a[1], s)), arrs.get((b[0], b[1], s))
        if x is None or y is None:
            continue
        idx = boot_idx(len(x)) if idx is None else idx
        nb, nc, p = v1.mcnemar_exact(x, y)
        d = (x[idx].mean(1) - y[idx].mean(1)) * 100
        per_seed[s] = {"a_pct": 100 * x.mean(), "b_pct": 100 * y.mean(), "diff_pp": 100 * (x.mean() - y.mean()),
                       "diff_ci95": ci(d), "a_only_tiles": nb, "b_only_tiles": nc, "mcnemar_exact_p": p}
        diffs.append((x, y))
    return summarise(per_seed, diffs, idx, lambda x, y, i: (x[i].mean(1) - y[i].mean(1)) * 100)


def summarise(per_seed, pairs, idx, stat_fn):
    if not per_seed:
        return None
    a = np.array([e["a_pct" if "a_pct" in e else "a"] for e in per_seed.values()])
    b = np.array([e["b_pct" if "b_pct" in e else "b"] for e in per_seed.values()])
    d = np.array([e["diff_pp" if "diff_pp" in e else "diff"] for e in per_seed.values()])
    avg = np.mean([stat_fn(x, y, idx) for x, y in pairs], axis=0)
    sd = lambda z: float(z.std(ddof=1)) if len(z) > 1 else None  # noqa: E731
    return {"per_seed": per_seed, "n_seeds": len(per_seed),
            "a_mean": float(a.mean()), "a_sd": sd(a), "b_mean": float(b.mean()), "b_sd": sd(b),
            "diff_mean": float(d.mean()), "diff_sd": sd(d), "seed_avg_diff_ci95": ci(avg),
            "same_sign_all_seeds": bool((d > 0).all() or (d < 0).all())}


def setA_rows(tbl, cell, s, c):
    """-> per-tile (tp, fp, n_gt) of one (condition, version) cell, seed and threshold, tile order."""
    x = tbl[(tbl["condition"] == cell[0]) & (tbl["seed"] == s) & (tbl["version"] == cell[1]) & np.isclose(tbl["conf"], c)]
    return x.sort_values("tile")[["tp", "fp", "n_gt"]].to_numpy(float)


def setA_comparison(tbl, ap, a, b, seeds, c, metric):
    per_seed, pairs, idx = {}, [], None
    for s in seeds:
        X, Y = setA_rows(tbl, a, s, c), setA_rows(tbl, b, s, c)
        if not len(X) or not len(Y):
            continue
        idx = boot_idx(len(X)) if idx is None else idx
        f = METRIC[metric]
        per_seed[s] = {"a": f(X.sum(0)), "b": f(Y.sum(0)), "diff": f(X.sum(0)) - f(Y.sum(0)),
                       "diff_ci95": ci(f(X[idx].sum(1).T) - f(Y[idx].sum(1).T))}
        if metric == "recall":
            ma, mb = f"M_{a[0][2:]}_s{s}", f"M_{b[0][2:]}_s{s}"
            per_seed[s]["ap50_a"], per_seed[s]["ap50_b"] = ap[ma][a[1]]["ap50"], ap[mb][b[1]]["ap50"]
        pairs.append((X, Y))
    return summarise(per_seed, pairs, idx, lambda X, Y, i: METRIC[metric](X[i].sum(1).T) - METRIC[metric](Y[i].sum(1).T))


def _p(s):
    tp, fp, _ = s
    return np.divide(tp, tp + fp, out=np.zeros_like(np.asarray(tp, float)), where=(tp + fp) > 0)


def _r(s):
    tp, _, gt = s
    return tp / gt


def _f1(s):
    p, r = _p(s), _r(s)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(np.asarray(p, float)), where=(p + r) > 0)


METRIC = {"precision": _p, "recall": _r, "f1": _f1}


# ------------------------------------------------- added after first results (not pre-registered)

def excess_difference(arrs, a, a0, b, b0, seeds):
    """(a - a0) - (b - b0) in points of tiles, paired by seed and tile: does one pipeline add more
    invention over its own detector's originals than the other? Same tile bootstrap as summarise."""
    per_seed, boots, idx = {}, [], None
    for s in seeds:
        xs = [arrs.get((c[0], c[1], s)) for c in (a, a0, b, b0)]
        if any(x is None for x in xs):
            continue
        idx = boot_idx(len(xs[0])) if idx is None else idx
        boot = 100 * ((xs[0][idx].mean(1) - xs[1][idx].mean(1)) - (xs[2][idx].mean(1) - xs[3][idx].mean(1)))
        point = 100 * ((xs[0].mean() - xs[1].mean()) - (xs[2].mean() - xs[3].mean()))
        per_seed[s] = {"diff_pp": float(point), "diff_ci95": ci(boot)}
        boots.append(boot)
    if not per_seed:
        return None
    d = np.array([e["diff_pp"] for e in per_seed.values()])
    return {"per_seed": per_seed, "n_seeds": len(d), "diff_mean": float(d.mean()),
            "diff_sd": float(d.std(ddof=1)) if len(d) > 1 else None, "seed_avg_diff_ci95": ci(np.mean(boots, axis=0)),
            "same_sign_all_seeds": bool((d > 0).all() or (d < 0).all())}


def pair_summary(pairs, value):
    if not pairs:
        return None
    d = np.array([e[value] for e in pairs.values()])
    excl = sum(1 for e in pairs.values() if e["diff_ci95"][0] > 0 or e["diff_ci95"][1] < 0)
    return {"pairs": pairs, "n_pairs": len(d), "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "min": float(d.min()), "max": float(d.max()), "n_ci_excluding_zero": excl}


def cross_seed_pairs(arrs, a, b, seeds):
    """Every (A seed, B seed) pairing, not only equal seeds, so the pairing cannot drive the result."""
    pairs, idx = {}, None
    for sa in seeds:
        for sb in seeds:
            x, y = arrs.get((a[0], a[1], sa)), arrs.get((b[0], b[1], sb))
            if x is None or y is None:
                continue
            idx = boot_idx(len(x)) if idx is None else idx
            _, _, p = v1.mcnemar_exact(x, y)
            pairs[f"{sa}v{sb}"] = {"diff_pp": 100 * (x.mean() - y.mean()),
                                   "diff_ci95": ci((x[idx].mean(1) - y[idx].mean(1)) * 100), "mcnemar_exact_p": p}
    return pair_summary(pairs, "diff_pp")


def cross_seed_recall(tbl, a, b, seeds, c):
    pairs, idx, f = {}, None, METRIC["recall"]
    for sa in seeds:
        for sb in seeds:
            X, Y = setA_rows(tbl, a, sa, c), setA_rows(tbl, b, sb, c)
            if not len(X) or not len(Y):
                continue
            idx = boot_idx(len(X)) if idx is None else idx
            pairs[f"{sa}v{sb}"] = {"diff": float(f(X.sum(0)) - f(Y.sum(0))),
                                   "diff_ci95": ci(f(X[idx].sum(1).T) - f(Y[idx].sum(1).T))}
    return pair_summary(pairs, "diff")


def majority_comparison(arrs, a, b, seeds):
    """A tile counts as invented when a strict majority of seeds (2 of 3) invent on it."""
    def vote(cell):
        xs = [arrs.get((cell[0], cell[1], s)) for s in seeds]
        return None if any(x is None for x in xs) else (2 * np.sum(xs, axis=0) > len(xs)).astype(int)
    x, y = vote(a), vote(b)
    if x is None or y is None:
        return None
    return binary_comparison({(*a, "majority"): x, (*b, "majority"): y}, a, b, ["majority"])


def added_after_first_results(bset, tbl, seeds):
    out = {"note": "added after first results; not pre-registered", "excess_over_own_originals": {},
           "excess_difference_MSR_minus_MBIC": {}, "cross_seed_pairs": {}, "majority_vote": {}}
    for key in PRIMARY[:2]:
        arrs = bset.get(key, {})
        for name, (a, b) in EXCESS.items():
            r = binary_comparison(arrs, a, b, [""] if a[0] == "v1" else seeds)
            if r:
                out["excess_over_own_originals"].setdefault(name, {})[key] = r
        r = excess_difference(arrs, *EXCESS["M_SR_on_sr_over_own_originals"],
                              *EXCESS["M_BIC_on_bicubic_over_own_originals"], seeds)
        if r:
            out["excess_difference_MSR_minus_MBIC"][key] = r
        for name, (a, b) in COMPARISONS.items():
            out["cross_seed_pairs"].setdefault(name, {})[key] = cross_seed_pairs(arrs, a, b, seeds)
            out["majority_vote"].setdefault(name, {})[key] = majority_comparison(arrs, a, b, seeds)
    for name, (a, b) in COMPARISONS.items():
        out["cross_seed_pairs"][name][PRIMARY[2]] = cross_seed_recall(tbl, a, b, seeds, 0.25)
    return out


def main():
    tbl = pd.read_csv(V2_RESULTS / "setA_per_tile.csv", dtype={"seed": str}, keep_default_na=False)
    ap = json.loads((V2_RESULTS / "setA_metrics.json").read_text())
    seeds = [str(s) for s in CFG["train"]["seeds"]]
    if not any(c.startswith("M_") for c in tbl["condition"].unique()):
        sys.exit("no v2 models in the results yet: train, then run 04, 05, 06 (and 07-09 for invention)")
    bset = setB_arrays(seeds)
    out = {"primary_endpoints": PRIMARY, "seeds": seeds, "comparisons": {}, "references": {}}
    for name, (a, b) in COMPARISONS.items():
        res = {}
        for key, arrs in bset.items():
            r = binary_comparison(arrs, a, b, seeds)
            if r:
                res[key] = r
        for c in CONFS:
            for m in METRIC:
                r = setA_comparison(tbl, ap, a, b, seeds, c, m)
                if r:
                    res[f"setA_{m}_conf{c}"] = r
        out["comparisons"][name] = res
    for cond, ver in REFERENCES + [("v1", "hr"), ("v1", "bicubic"), ("v1", "sr")]:
        ref = {}
        for key, arrs in bset.items():
            vals = [100 * v.mean() for (cc, vv, s), v in arrs.items() if cc == cond and vv == ver]
            if vals:
                ref[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                            "per_seed": vals}
        out["references"][f"{cond}_on_{ver}"] = ref
    out["added_after_first_results"] = added_after_first_results(bset, tbl, seeds)
    (V2_RESULTS / "stats.json").write_text(json.dumps(out, indent=2, default=float))
    for name, res in out["comparisons"].items():
        print(f"\n{name}")
        for key in PRIMARY:
            if key in res:
                e = res[key]
                print(f"  {key:40s} A {e['a_mean']:7.3f}  B {e['b_mean']:7.3f}  diff {e['diff_mean']:+7.3f} "
                      f"(seed-avg CI {e['seed_avg_diff_ci95'][0]:+.3f}, {e['seed_avg_diff_ci95'][1]:+.3f})  "
                      f"same sign all seeds: {e['same_sign_all_seeds']}")
    post = out["added_after_first_results"]
    print("\nadded after first results")
    for name, res in post["excess_over_own_originals"].items():
        e = res[PRIMARY[0]]
        print(f"  excess {name:40s} {e['diff_mean']:+6.1f} pp (CI {e['seed_avg_diff_ci95'][0]:+.1f}, "
              f"{e['seed_avg_diff_ci95'][1]:+.1f})")
    for name, res in post["cross_seed_pairs"].items():
        e = res[PRIMARY[0]]
        if e:
            print(f"  {name} 9 pairs, invention: {e['n_negative']} below / {e['n_positive']} above zero, "
                  f"range {e['min']:+.1f} to {e['max']:+.1f}")
    print(f"\n-> {V2_RESULTS / 'stats.json'}")


if __name__ == "__main__":
    main()
