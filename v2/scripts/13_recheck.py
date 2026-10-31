"""Added after first results: blind test-retest re-check of the audit verdicts.

Compares the re-check pass (08_label.py --recheck: v2/results/audit/audit_labels_recheck.csv, with the
first-pass verdicts in audit_recheck_key.csv) against the first pass:

  agreement  Cohen's kappa, raw agreement and the old x new confusion matrix, for all re-checked
             crops and per source (primary-cell sample, originals queue). The re-check rule (any visible
             hull inside the box = ship) is looser than the first pass's, so disagreement mixes a rule
             change with rater noise.
  impact     invention at the audit threshold, all tiles, both bands, for every primary cell and seed,
             and the excess-over-originals rows, if the re-check verdicts replaced the first-pass ones.
             Only re-checked crops change; every other crop keeps its first-pass verdict.

GATE: with nothing replaced, the recomputed invention must equal setB_invention.json for every cell
it covers, so this script counts exactly as 09 does. The first pass stays the main result; when kappa
is below audit.recheck.min_kappa the script still writes v2/results/recheck.json but exits with
status 2, so nothing downstream changes before the re-check has been reviewed.
"""
import csv, importlib.util, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2, V2_RESULTS, load_v1_script, read_ids  # noqa: E402

AUD = V2_RESULTS / "audit"
RC = CFG["audit"]["recheck"]
CONF_KEY = f"conf{CFG['audit']['conf']}"
VERDICTS = ("ship", "unsure", "not_ship")
SEEDS = [str(s) for s in CFG["train"]["seeds"]]
PRIMARY_CELLS = [tuple(c) for c in CFG["audit"]["priority_cells"]]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, V2 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def passes():
    """-> [{crop_id, source, old_verdict, new_verdict}]; exits if any re-check verdict is blank or invalid."""
    key = list(csv.DictReader(open(AUD / "audit_recheck_key.csv")))
    new = {r["crop_id"]: (r["verdict"] or "").strip() for r in csv.DictReader(open(AUD / "audit_labels_recheck.csv"))}
    blank = [r["crop_id"] for r in key if not new.get(r["crop_id"])]
    if blank:
        sys.exit(f"{len(blank)} of {len(key)} re-check crops unlabelled: .venv/bin/python v2/scripts/08_label.py --recheck")
    bad = sorted({v for v in new.values() if v not in VERDICTS})
    if bad:
        sys.exit(f"INVALID re-check verdicts: {bad}")
    return [{**r, "new_verdict": new[r["crop_id"]]} for r in key]


def agreement(rows, kappa):
    old, new = [r["old_verdict"] for r in rows], [r["new_verdict"] for r in rows]
    matrix = {o: {n: sum(1 for a, b in zip(old, new) if (a, b) == (o, n)) for n in VERDICTS} for o in VERDICTS}
    return {"n": len(rows), "kappa": kappa(old, new), "raw_agreement": float(np.mean([a == b for a, b in zip(old, new)])),
            "confusion_old_rows_new_cols": matrix}


def audited_cells(inv_json):
    """-> {(condition, version, seed): 09's entry} for every cell 09 counted."""
    return {(res["condition"], ver, res["seed"]): e for model, res in inv_json.items() if not model.startswith("_")
            for ver, e in res.items() if isinstance(e, dict)}


def invention(dets, ids, bands, cells):
    """-> {band: {cell: 0/1 over tiles}} for the audited cells, counted as 09 counts (conf > audit conf)."""
    at = {t: i for i, t in enumerate(ids)}
    out = {band: {c: np.zeros(len(ids), int) for c in cells} for band in bands}
    for d in dets:
        cell = (d["condition"], d["version"], d["seed"])
        if cell not in cells or d["conf"] <= CFG["audit"]["conf"]:
            continue
        for band, inv in bands.items():
            out[band][cell][at[d["tile"]]] |= d["verdict"] in inv
    return out


def gate(arrs, cells):
    """Recomputed first-pass tile counts must equal 09's for every audited cell."""
    return [f"{cell} {band}: {int(a[cell].sum())} vs 09's {e[CONF_KEY][band]['all']['tiles']}"
            for band, a in arrs.items() for cell, e in cells.items()
            if int(a[cell].sum()) != e[CONF_KEY][band]["all"]["tiles"]]


def cell_impact(old, new):
    out = {}
    for band in old:
        for cond, ver in PRIMARY_CELLS:
            o = [100 * old[band][(cond, ver, s)].mean() for s in SEEDS]
            n = [100 * new[band][(cond, ver, s)].mean() for s in SEEDS]
            out.setdefault(f"{cond}_on_{ver}", {})[band] = {
                "old_per_seed": o, "new_per_seed": n, "old_mean": float(np.mean(o)), "new_mean": float(np.mean(n)),
                "change_mean": float(np.mean(n) - np.mean(o))}
    return out


def pick(e):
    return {"diff_mean": e["diff_mean"], "ci95": e["seed_avg_diff_ci95"], "same_sign_all_seeds": e["same_sign_all_seeds"]}


def excess_impact(old, new, m10):
    out = {}
    for band in old:
        for name, (a, b) in m10.EXCESS.items():
            seeds = [""] if a[0] == "v1" else SEEDS
            o, n = m10.binary_comparison(old[band], a, b, seeds), m10.binary_comparison(new[band], a, b, seeds)
            out.setdefault(name, {})[band] = {"old": pick(o), "new": pick(n)}
        args = (*m10.EXCESS["M_SR_on_sr_over_own_originals"], *m10.EXCESS["M_BIC_on_bicubic_over_own_originals"], SEEDS)
        out.setdefault("excess_difference_MSR_minus_MBIC", {})[band] = {
            "old": pick(m10.excess_difference(old[band], *args)), "new": pick(m10.excess_difference(new[band], *args))}
    return out


def main():
    m09, m10 = load_script("09_recount_invention"), load_script("10_stats")
    rows = passes()
    cv, v1key, v1final, _ = m09.verdicts(True)
    crop_of = {r["cluster"]: r["crop_id"] for r in csv.DictReader(open(AUD / "clusters.csv"))}
    new_v = {r["crop_id"]: r["new_verdict"] for r in rows}
    cv_new = {k: new_v.get(crop_of[k], v) for k, v in cv.items()}
    ids, cells = read_ids("setB"), audited_cells(json.loads((V2_RESULTS / "setB_invention.json").read_text()))
    old = invention(m09.detection_rows(cv, v1key, v1final, False), ids, m09.BANDS, cells)
    new = invention(m09.detection_rows(cv_new, v1key, v1final, False), ids, m09.BANDS, cells)
    bad = gate(old, cells)
    if bad:
        sys.exit("GATE FAILED, this script does not count as 09 does:\n  " + "\n  ".join(bad[:10]))

    strata = load_script("08_label").primary_strata(load_v1_script("11_label"))
    primary = [r for r in rows if r["source"] == "primary"]
    sample = {"primary_stratum_sizes": {v: len(c) for v, c in strata.items()},
              "primary_taken": {v: sum(1 for r in primary if r["old_verdict"] == v) for v in VERDICTS},
              "originals_queue": sum(1 for r in rows if r["source"] == "originals_queue")}
    by_source = {s: [r for r in rows if r["source"] == s] for s in sorted({r["source"] for r in rows})}
    agree = {"all": agreement(rows, m09.kappa), **{s: agreement(rs, m09.kappa) for s, rs in by_source.items()}}
    passed = agree["all"]["kappa"] >= RC["min_kappa"]
    out = {"note": "added after first results; the first pass stays the main result", "rule": RC["rule"],
           "min_kappa": RC["min_kappa"], "kappa_passed": passed, "sample": sample, "agreement": agree,
           "n_verdicts_changed": sum(1 for r in rows if r["old_verdict"] != r["new_verdict"]),
           "impact_primary_cells": cell_impact(old, new), "impact_excess": excess_impact(old, new, m10)}
    (V2_RESULTS / "recheck.json").write_text(json.dumps(out, indent=2, default=float))

    for name, a in agree.items():
        print(f"{name:16s} n {a['n']:3d}  kappa {a['kappa']:.3f}  raw agreement {a['raw_agreement']:.3f}")
    print("confusion (rows = first pass, cols = re-check):", json.dumps(agree["all"]["confusion_old_rows_new_cols"]))
    for cell, e in out["impact_primary_cells"].items():
        print(f"  {cell:16s} " + "  ".join(f"{b} {v['old_mean']:5.1f} -> {v['new_mean']:5.1f}" for b, v in e.items()))
    for name, e in out["impact_excess"].items():
        print(f"  {name:38s} " + "  ".join(f"{b} {v['old']['diff_mean']:+5.1f} -> {v['new']['diff_mean']:+5.1f}"
                                           for b, v in e.items()))
    print(f"-> {V2_RESULTS / 'recheck.json'}")
    if not passed:
        print(f"\nSTOP: kappa {agree['all']['kappa']:.3f} < {RC['min_kappa']}. The first pass stays the main result; "
              "review the re-check before changing anything.")
        sys.exit(2)


if __name__ == "__main__":
    main()
