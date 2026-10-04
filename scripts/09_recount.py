"""Recompute Set B from the blinded manual audit.

Joins results/audit_key.csv (crop_id -> version, tile, conf, box) with results/audit_labels.csv
(crop_id -> verdict). Verdicts are `ship` / `not_ship` / `unsure`:

  ship      a real vessel is present in the HR crop. DOTA never labelled it and the HR detector
            missed it, so it was never an invention -- it is a recovery. Not counted.
  not_ship  nothing vessel-like is there (rooftop, car, dock, wave, empty water). Counted as
            invented in BOTH bands.
  unsure    counted as invented in the UPPER band only.

So:  lower bound = not_ship                 (fewest inventions)
     upper bound = not_ship + unsure        (most inventions)

Both SR and bicubic are audited, so the paired comparison stays symmetric. A blank verdict is a
hard error: a silent default would quietly manufacture a result.
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, DATA, RESULTS, bootstrap_pct_ci, bootstrap_diff_ci, mcnemar_exact  # noqa: E402

VALID = {"ship", "not_ship", "unsure"}
KEY = RESULTS / "audit_key.csv"


def load(labels_path):
    key = {r["crop_id"]: r for r in csv.DictReader(open(KEY))}
    labs = {r["crop_id"]: (r.get("verdict") or "").strip().lower()
            for r in csv.DictReader(open(labels_path))}

    missing = sorted(set(key) - set(labs))
    extra = sorted(set(labs) - set(key))
    if missing or extra:
        sys.exit(f"KEY/LABELS MISMATCH: {len(missing)} crop_ids missing from labels "
                 f"{missing[:5]}, {len(extra)} unknown ids in labels {extra[:5]}")

    blank = sorted(c for c, v in labs.items() if not v)
    bad = sorted((c, v) for c, v in labs.items() if v and v not in VALID)
    if bad:
        print("INVALID verdicts (must be ship / not_ship / unsure):", file=sys.stderr)
        for c, v in bad[:20]:
            print(f"  {c}: {v!r}", file=sys.stderr)
        sys.exit(1)
    if blank:
        print(f"UNLABELLED: {len(blank)}/{len(key)} crops have no verdict.", file=sys.stderr)
        print(f"  first few: {', '.join(blank[:10])}", file=sys.stderr)
        print("  Refusing to recount. A blank is not a verdict; treating it as one would "
              "manufacture a result.", file=sys.stderr)
        print("  Run: .venv/bin/python scripts/11_label.py", file=sys.stderr)
        sys.exit(1)

    rows = []
    for c, k in key.items():
        rows.append({"crop_id": c, "version": k["version"], "tile": k["tile"],
                     "conf": float(k["conf"]), "verdict": labs[c]})
    return rows


def recount(rows, tiles, rule, conf_min):
    per = {v: {t: 0 for t in tiles} for v in ("sr", "bicubic")}
    for r in rows:
        if r["conf"] < conf_min:
            continue
        v = r["verdict"]
        invented = (v == "not_ship") or (rule == "upper" and v == "unsure")
        if invented:
            per[r["version"]][r["tile"]] += 1
    return per


def block(per, tiles, label):
    out, hits = {"label": label}, {}
    for v in ("sr", "bicubic"):
        h = [1 if per[v][t] > 0 else 0 for t in tiles]
        hits[v] = h
        lo, hi = bootstrap_pct_ci(h)
        out[v] = {"tiles_with_ship": sum(h), "pct_tiles": 100.0 * sum(h) / len(h),
                  "pct_ci95": [lo, hi], "total_ships": sum(per[v].values()),
                  "per_100_tiles": 100.0 * sum(per[v].values()) / len(tiles)}
    b, c, p = mcnemar_exact(hits["sr"], hits["bicubic"])
    dlo, dhi = bootstrap_diff_ci(hits["sr"], hits["bicubic"])
    out["paired"] = {"sr_only_tiles": b, "bicubic_only_tiles": c, "mcnemar_exact_p": p,
                     "diff_pct_points": out["sr"]["pct_tiles"] - out["bicubic"]["pct_tiles"],
                     "diff_ci95": [dlo, dhi]}
    return out


def hr_counts(conf_min):
    """Ship detections on the ORIGINAL imagery, per tile, at this confidence."""
    out = {}
    for r in csv.DictReader(open(RESULTS / "setB_counts.csv")):
        if r["version"] == "hr" and float(r["conf"]) == conf_min:
            out[r["tile"]] = int(r["n_ships"])
    return out


def beyond_original(rows, tiles, conf_min):
    """(b) CONFIDENCE BEYOND THE ORIGINAL.

    % of tiles where the detector fired on this version but did NOT fire on HR -- regardless of
    whether the box later turned out to be a real vessel. Then the verdict split of those boxes:
    how often that extra confidence was justified.

    SR is computed from a 4x-degraded copy of HR, so by the data processing inequality it cannot
    carry more information about the scene than HR does. A detection present on SR and absent on HR
    is therefore not new evidence; it is synthesized texture, or the same evidence re-formatted into
    something the detector's priors accept."""
    hr = hr_counts(conf_min)
    assert set(hr) >= set(tiles), "setB_counts.csv is missing HR rows for some tiles"
    out, hits = {"hr_tiles_fired": sum(1 for t in tiles if hr[t] > 0)}, {}
    for v in ("sr", "bicubic"):
        fired = {t: 0 for t in tiles}
        verd = defaultdict(int)
        for r in rows:
            if r["version"] != v or r["conf"] < conf_min:
                continue
            fired[r["tile"]] += 1
        for r in rows:
            if r["version"] != v or r["conf"] < conf_min or hr[r["tile"]] > 0:
                continue
            verd[r["verdict"]] += 1
        h = [1 if (fired[t] > 0 and hr[t] == 0) else 0 for t in tiles]
        hits[v] = h
        lo, hi = bootstrap_pct_ci(h)
        n_box = sum(verd.values())
        out[v] = {
            "tiles_fired_beyond_hr": sum(h),
            "pct_tiles": 100.0 * sum(h) / len(h),
            "pct_ci95": [lo, hi],
            "boxes_on_those_tiles": n_box,
            "verdict_counts": dict(verd),
            "verdict_share_pct": {k: 100.0 * verd[k] / n_box for k in verd} if n_box else {},
        }
    b, c, p = mcnemar_exact(hits["sr"], hits["bicubic"])
    dlo, dhi = bootstrap_diff_ci(hits["sr"], hits["bicubic"])
    out["paired"] = {"sr_only_tiles": b, "bicubic_only_tiles": c, "mcnemar_exact_p": p,
                     "diff_pct_points": out["sr"]["pct_tiles"] - out["bicubic"]["pct_tiles"],
                     "diff_ci95": [dlo, dhi]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(RESULTS / "audit_labels.csv"))
    ap.add_argument("--out", default=str(RESULTS / "setB_audited.json"))
    ap.add_argument("--exclude", default="", help="comma-separated tile ids to drop entirely "
                                                 "(sensitivity analysis; the tile leaves the "
                                                 "denominator, not just the numerator)")
    args = ap.parse_args()

    rows = load(args.labels)
    try:                       # keep the json portable: no absolute paths
        lab_rel = str(Path(args.labels).resolve().relative_to(ROOT))
    except ValueError:
        lab_rel = args.labels
    tally = defaultdict(int)
    for r in rows:
        tally[(r["version"], r["verdict"])] += 1
    print(f"{len(rows)}/{len(rows)} crops labelled\n\nverdict tally:")
    for k in sorted(tally):
        print(f"  {k[0]:8s} {k[1]:9s} {tally[k]}")

    tiles = sorted({p.stem for p in (DATA / "versions" / "hr" / "setB" / "images").glob("*.png")})
    drop = {t.strip() for t in args.exclude.split(",") if t.strip()}
    if drop:
        unknown = drop - set(tiles)
        if unknown:
            sys.exit(f"--exclude names tiles that are not in Set B: {sorted(unknown)}")
        tiles = [t for t in tiles if t not in drop]
        rows = [r for r in rows if r["tile"] not in drop]
        tally = defaultdict(int)
        for r in rows:
            tally[(r["version"], r["verdict"])] += 1
        print(f"\nEXCLUDING {len(drop)} tile(s): {sorted(drop)}")
        print(f"  -> {len(tiles)} tiles, {len(rows)} crops remain")
        for k in sorted(tally):
            print(f"  {k[0]:8s} {k[1]:9s} {tally[k]}")
    out = {"n_tiles": len(tiles), "n_crops": len(rows), "labels_file": lab_rel,
           "excluded_tiles": sorted(drop),
           "verdict_tally": {f"{k[0]}_{k[1]}": v for k, v in tally.items()}, "bands": {}}
    for conf_min in (0.25, 0.5):
        for rule, lab in (("lower", "unsure counted as a real ship (lower bound on invention)"),
                          ("upper", "unsure counted as invented (upper bound on invention)")):
            out["bands"][f"conf{conf_min}_{rule}"] = block(
                recount(rows, tiles, rule, conf_min), tiles, lab)

    out["beyond_original"] = {f"conf{c}": beyond_original(rows, tiles, c) for c in (0.25, 0.5)}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n(b) CONFIDENCE BEYOND THE ORIGINAL — fired on the version, silent on HR")
    print(f"{'conf':>5s} {'version':9s} {'tiles':>6s} {'% tiles':>8s} {'95% CI':>14s} "
          f"{'boxes':>6s}  verdict split")
    for ck, bo in out["beyond_original"].items():
        for v in ("sr", "bicubic"):
            e = bo[v]
            sp = "  ".join(f"{k}={e['verdict_counts'].get(k,0)}"
                           for k in ("ship", "not_ship", "unsure"))
            print(f"{ck[4:]:>5s} {v:9s} {e['tiles_fired_beyond_hr']:6d} {e['pct_tiles']:7.1f}% "
                  f"[{e['pct_ci95'][0]:5.1f},{e['pct_ci95'][1]:5.1f}] {e['boxes_on_those_tiles']:6d}  {sp}")
    print(f"\n{'band':26s} {'SR tiles':>9s} {'SR ships':>9s} {'bic tiles':>10s} {'bic ships':>10s} "
          f"{'diff pp':>8s} {'McNemar p':>11s}")
    for k, b in out["bands"].items():
        print(f"{k:26s} {b['sr']['tiles_with_ship']:9d} {b['sr']['total_ships']:9d} "
              f"{b['bicubic']['tiles_with_ship']:10d} {b['bicubic']['total_ships']:10d} "
              f"{b['paired']['diff_pct_points']:+8.1f} {b['paired']['mcnemar_exact_p']:11.2e}")
    print(f"\n-> {args.out}   then rerun scripts/06_summary.py")


if __name__ == "__main__":
    main()
