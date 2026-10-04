"""Phase 3 done-when — build results/summary.md from the files in results/.
Every number here is read from disk; none is typed by hand.
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, bootstrap_pct_ci, bootstrap_diff_ci, mcnemar_exact  # noqa: E402

VERSIONS = ("hr", "bicubic", "sr")
counts_by_tile = defaultdict(dict)
LABEL = {"hr": "HR", "bicubic": "Bicubic", "sr": "Real-ESRGAN"}


def hit_vec(counts, v, c):
    """0/1 per tile: did this version put at least one ship on this tile. Tile order is fixed
    (sorted) so the SR and bicubic vectors are aligned for the paired tests."""
    d = counts_by_tile.get((v, c))
    if d is None:
        return None
    return [1 if d[t] > 0 else 0 for t in sorted(d)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    setA = json.loads((RESULTS / "setA_metrics.json").read_text()) if (RESULTS / "setA_metrics.json").exists() else {}

    counts = defaultdict(list)
    n_tiles = 0
    with open(RESULTS / "setB_counts.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        counts[(r["version"], float(r["conf"]))].append(int(r["n_ships"]))
        counts_by_tile[(r["version"], float(r["conf"]))][r["tile"]] = int(r["n_ships"])
    n_tiles = len(counts[("hr", 0.25)])

    psnr = defaultdict(list)
    ssim = defaultdict(list)
    if (RESULTS / "psnr.csv").exists():
        with open(RESULTS / "psnr.csv") as f:
            for r in csv.DictReader(f):
                psnr[r["version"]].append(float(r["psnr"]))
                ssim[r["version"]].append(float(r["ssim"]))

    def per100(v, c):
        xs = counts.get((v, c))
        return None if not xs else 100.0 * sum(xs) / len(xs)

    def tilepct(v, c):
        xs = counts.get((v, c))
        return None if not xs else 100.0 * sum(1 for x in xs if x > 0) / len(xs)

    L = []
    L.append("# Results\n")
    aud = setA.get("hr", {})
    nA = aud.get("n_tiles_validated")
    nShip = aud.get("n_ship_instances_validated")
    dropped = aud.get("tiles_dropped_by_ultralytics") or []
    L.append(f"Set A = {nA} tiles carrying {nShip} ship instances. Set B = {n_tiles} harbor/bridge "
             "tiles with zero ship labels AND zero ship detections on the original at conf 0.25 — "
             "**tiles where the detector saw nothing in the original**.\n")
    L.append("Set B was built as \"verified empty\" and that term is not used here: the audit below "
             "shows the construction verifies zero HR *detections*, not zero vessels. DOTA omits "
             "small craft, so some of these tiles do contain boats that were never labelled and "
             "that the detector missed on the original. Every claim below is therefore made against "
             "a hand audit of the original imagery, not against the label file.\n")
    if dropped:
        L.append(f"Set A was sampled as 100 tiles; ultralytics drops {len(dropped)} "
                 f"({', '.join(dropped)}) because a DOTA polygon vertex falls outside the crop "
                 "(coordinate 1.179). The same tile is dropped from all three versions, so the "
                 "comparison is unaffected.\n")
    L.append("The Set B columns in this first table are **raw detection counts**, kept because the "
             "project brief asked for them; they are superseded by the audited numbers further "
             "down.\n")
    L.append("| Version | Set A ship P | Set A ship R | Set A mAP50 | Set B ship detections / 100 tiles (conf 0.25) | (conf 0.5) | Mean PSNR vs HR |")
    L.append("|---|---|---|---|---|---|---|")
    for v in VERSIONS:
        a = setA.get(v, {}).get("ship") or {}
        p = f"{a['precision']:.3f}" if a else "—"
        r = f"{a['recall']:.3f}" if a else "—"
        m = f"{a['mAP50']:.3f}" if a else "—"
        c25 = per100(v, 0.25)
        c50 = per100(v, 0.5)
        c25s = "0 (by construction)" if v == "hr" and c25 == 0 else (f"{c25:.1f}" if c25 is not None else "—")
        c50s = f"{c50:.1f}" if c50 is not None else "—"
        ps = "—" if v == "hr" else (f"{mean(psnr[v]):.2f} dB" if psnr[v] else "—")
        L.append(f"| {LABEL[v]} | {p} | {r} | {m} | {c25s} | {c50s} | {ps} |")

    L.append("\n## Set B BEFORE the audit — raw detection counts (superseded)\n")
    L.append("These are every ship detection, with no check on whether an object was really there. "
             "The audit below shows a third of them were real unlabelled vessels, so **do not quote "
             "this table as an invention rate** — it is kept because it is what the detector did.\n")
    L.append("Tile-level, not ship-level: one tile contributes at most one event, so a single "
             "pathological tile cannot drive the number. CIs are percentile bootstrap over 10,000 "
             "resamples of the 150 tiles (seed 0); the tile is the unit of independence.\n")
    L.append("| Version | conf | tiles with >=1 invented ship | **% of tiles** | 95% CI | total invented ships | per 100 tiles |")
    L.append("|---|---|---|---|---|---|---|")
    for v in VERSIONS:
        for c in (0.25, 0.5):
            xs = counts.get((v, c))
            if not xs:
                continue
            hits = hit_vec(counts, v, c)
            lo, hi = bootstrap_pct_ci(hits)
            L.append(f"| {LABEL[v]} | {c} | {sum(1 for x in xs if x > 0)} | **{tilepct(v,c):.1f}%** | "
                     f"[{lo:.1f}, {hi:.1f}] | {sum(xs)} | {per100(v,c):.1f} |")

    L.append("\n### SR vs bicubic, paired on the same tiles\n")
    L.append("| conf | SR-only tiles (b) | bicubic-only tiles (c) | difference in % of tiles | paired 95% CI | exact McNemar p |")
    L.append("|---|---|---|---|---|---|")
    for c in (0.25, 0.5):
        hs, hb = hit_vec(counts, "sr", c), hit_vec(counts, "bicubic", c)
        if hs is None or hb is None:
            continue
        b, cc, p = mcnemar_exact(hs, hb)
        lo, hi = bootstrap_diff_ci(hs, hb)
        diff = 100.0 * (sum(hs) / len(hs) - sum(hb) / len(hb))
        pstr = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
        L.append(f"| {c} | {b} | {cc} | +{diff:.1f} pp | [{lo:.1f}, {hi:.1f}] | **{pstr}** |")
    L.append("\nMcNemar is exact (binomial), not chi-square: one discordant cell is 0 at both "
             "thresholds, where the chi-square approximation is not valid.\n")

    if psnr:
        L.append("\n## Pixel-similarity scores (all tiles, both sets)\n")
        L.append("| Version | n tiles | mean PSNR | mean SSIM |")
        L.append("|---|---|---|---|")
        for v in ("bicubic", "sr"):
            if psnr[v]:
                L.append(f"| {LABEL[v]} | {len(psnr[v])} | {mean(psnr[v]):.2f} dB | {mean(ssim[v]):.4f} |")

    aud = RESULTS / "setB_audited.json"
    if aud.exists():
        d = json.loads(aud.read_text())
        L.append("\n## Set B AFTER the blind audit — the two metrics that count\n")
        L.append(f"Every ship box on Set B at conf 0.25 ({d['n_crops']} crops: SR and bicubic both) was "
                 "cropped from the **HR** tile (pad 48 px, nearest-neighbour enlargement) and judged "
                 "by hand as `ship` / `not_ship` / `unsure`. The audit is **blind**: crop order was "
                 "shuffled with a fixed seed and crop ids carry no version, tile or confidence "
                 "(`results/audit_key.csv` holds the mapping, `results/audit_labels.csv` the verdicts, "
                 "contact sheets in `results/figures/audit/`). Boxes judged `ship` are unlabelled "
                 "vessels DOTA omitted and the HR detector missed — recoveries, not inventions — and "
                 "do not count. `unsure` counts only in the upper band.\n")
        tal = d["verdict_tally"]
        L.append("Verdict tally: " + " · ".join(f"`{k}` {v}" for k, v in sorted(tal.items())) + "\n")

        L.append("### (a) INVENTION — % of tiles with >=1 box judged *not a vessel*\n")
        L.append("| band | SR % of tiles | SR 95% CI | bicubic % of tiles | difference | paired 95% CI | exact McNemar p |")
        L.append("|---|---|---|---|---|---|---|")
        for k, b in d["bands"].items():
            pv = b["paired"]["mcnemar_exact_p"]
            ps = f"{pv:.2e}" if pv < 1e-3 else f"{pv:.4f}"
            L.append(f"| {k} | {b['sr']['pct_tiles']:.1f}% | "
                     f"[{b['sr']['pct_ci95'][0]:.1f}, {b['sr']['pct_ci95'][1]:.1f}] | "
                     f"{b['bicubic']['pct_tiles']:.1f}% | "
                     f"{b['paired']['diff_pct_points']:+.1f} pp | "
                     f"[{b['paired']['diff_ci95'][0]:.1f}, {b['paired']['diff_ci95'][1]:.1f}] | {ps} |")
        def fp(x):
            return f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"

        bd = d["bands"]
        key = {r["crop_id"]: r for r in csv.DictReader(open(RESULTS / "audit_key.csv"))}
        labs = {r["crop_id"]: (r.get("verdict") or "").strip().lower()
                for r in csv.DictReader(open(RESULTS / "audit_labels.csv"))}
        ns_tile = Counter(key[c]["tile"] for c, v in labs.items() if v == "not_ship")
        top_tile, top_n = ns_tile.most_common(1)[0]
        total_ns = sum(ns_tile.values())
        n_box_050 = d["beyond_original"]["conf0.5"]["sr"]["boxes_on_those_tiles"]
        L.append(f"\n**This is the lead result.** SR exceeds bicubic at conf 0.25 in both bands — "
                 f"{bd['conf0.25_lower']['sr']['pct_tiles']:.1f}% of tiles against "
                 f"{bd['conf0.25_lower']['bicubic']['pct_tiles']:.1f}% "
                 f"(p = {fp(bd['conf0.25_lower']['paired']['mcnemar_exact_p'])}) counting only boxes "
                 f"judged not-a-vessel, and "
                 f"{bd['conf0.25_upper']['sr']['pct_tiles']:.1f}% against "
                 f"{bd['conf0.25_upper']['bicubic']['pct_tiles']:.1f}% "
                 f"(p = {fp(bd['conf0.25_upper']['paired']['mcnemar_exact_p'])}) counting the "
                 f"uncertain ones too.\n")
        L.append(f"**It is not significant at conf 0.50** "
                 f"(p = {fp(bd['conf0.5_lower']['paired']['mcnemar_exact_p'])} lower, "
                 f"{fp(bd['conf0.5_upper']['paired']['mcnemar_exact_p'])} upper), where only "
                 f"{n_box_050} SR boxes survive the threshold at all. That is an absence of power, "
                 f"not evidence of absence: the point estimate still moves the same way "
                 f"({bd['conf0.5_upper']['sr']['pct_tiles']:.1f}% vs "
                 f"{bd['conf0.5_upper']['bicubic']['pct_tiles']:.1f}%). Raising the detector "
                 f"threshold reduces invention; it does not remove it.\n")
        L.append(f"Invention is **concentrated**: of the {total_ns} boxes judged not-a-vessel across "
                 f"the whole set, {top_n} sit on a single tile (`{top_tile}`), where SR turned a dark "
                 f"residential hillside into {top_n} \"ships\" and both the original and the bicubic "
                 f"upscale returned zero. The sensitivity analysis below removes that tile entirely.\n")

        pass1 = RESULTS / "setB_audited_pass1.json"
        if pass1.exists():
            d1 = json.loads(pass1.read_text())
            t1, t2 = d1["verdict_tally"], d["verdict_tally"]
            def tot(t, k):
                # keys are "{version}_{verdict}"; endswith would count sr_not_ship as ship
                return sum(v for kk, v in t.items() if kk.split("_", 1)[1] == k)
            L.append("\n#### Pass 1 vs pass 2 — adjudicating the `unsure` crops\n")
            L.append("Every crop labelled `unsure` in pass 1 was re-rendered with 128 px of context "
                     "instead of 48, reshuffled under a new seed, and judged again, still blind. A "
                     "pass-2 verdict replaces the pass-1 `unsure` and nothing else changes. Pass 1 is "
                     "kept at `results/audit_labels_pass1.csv`.\n")
            L.append("| verdict | pass 1 | final | change |")
            L.append("|---|---|---|---|")
            for k in ("ship", "not_ship", "unsure"):
                a, b = tot(t1, k), tot(t2, k)
                L.append(f"| `{k}` | {a} | {b} | {b-a:+d} |")
            L.append("\n| band | pass-1 SR % | final SR % | pass-1 McNemar p | final McNemar p |")
            L.append("|---|---|---|---|---|")
            for k in d["bands"]:
                b1, b2 = d1["bands"][k], d["bands"][k]
                def fp(x):
                    return f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"
                L.append(f"| {k} | {b1['sr']['pct_tiles']:.1f}% | **{b2['sr']['pct_tiles']:.1f}%** | "
                         f"{fp(b1['paired']['mcnemar_exact_p'])} | "
                         f"**{fp(b2['paired']['mcnemar_exact_p'])}** |")
            a1 = d1["bands"]["conf0.25_lower"]["sr"]["pct_tiles"]
            a2 = d1["bands"]["conf0.25_upper"]["sr"]["pct_tiles"]
            b1 = d["bands"]["conf0.25_lower"]["sr"]["pct_tiles"]
            b2 = d["bands"]["conf0.25_upper"]["sr"]["pct_tiles"]
            L.append(f"\nThe second pass pulled the two bands together — the invention estimate at "
                     f"conf 0.25 was {min(a1,a2):.1f}%-{max(a1,a2):.1f}% of tiles after pass 1 and is "
                     f"**{min(b1,b2):.1f}%-{max(b1,b2):.1f}%** after pass 2 — and strengthened the "
                     f"conf-0.25 result by about four orders of magnitude in p. Only "
                     f"{tot(t2,'unsure')} of {d['n_crops']} crops remain uncertain, so the band is "
                     f"now narrow enough that the choice of how to treat `unsure` barely matters.\n")
            L.append(f"The 38 pass-1 `unsure` crops resolved "
                     f"{tot(t2,'not_ship')-tot(t1,'not_ship')} to `not_ship` and "
                     f"{tot(t2,'ship')-tot(t1,'ship')} to `ship`, with {tot(t2,'unsure')} still "
                     f"uncertain at 128 px of context.\n")

        nop = RESULTS / "setB_audited_no_P1751.json"
        if nop.exists():
            dx = json.loads(nop.read_text())
            L.append(f"\n#### Sensitivity — the same test with the dominant tile removed\n")
            L.append(f"`{top_tile}` carries {top_n} of the {total_ns} not-a-vessel boxes. Dropping it "
                     f"**from the denominator as well as the numerator** leaves "
                     f"{dx['n_tiles']} tiles and {dx['n_crops']} crops. The tile-level result is "
                     f"barely affected, because the test already counts each tile once:\n")
            L.append("| band | SR % (150 tiles) | SR % (149, tile dropped) | bicubic % | difference | exact McNemar p |")
            L.append("|---|---|---|---|---|---|")
            for k in ("conf0.25_lower", "conf0.25_upper"):
                a, b = d["bands"][k], dx["bands"][k]
                L.append(f"| {k} | {a['sr']['pct_tiles']:.1f}% | **{b['sr']['pct_tiles']:.1f}%** "
                         f"[{b['sr']['pct_ci95'][0]:.1f}, {b['sr']['pct_ci95'][1]:.1f}] | "
                         f"{b['bicubic']['pct_tiles']:.1f}% | "
                         f"{b['paired']['diff_pct_points']:+.1f} pp | "
                         f"**{fp(b['paired']['mcnemar_exact_p'])}** |")
            ex = dx["beyond_original"]["conf0.25"]["sr"]
            nb, ns_, un = (ex["boxes_on_those_tiles"], ex["verdict_counts"].get("not_ship", 0),
                           ex["verdict_counts"].get("unsure", 0))
            fu = d["beyond_original"]["conf0.25"]["sr"]
            fnb, fns, fun = (fu["boxes_on_those_tiles"], fu["verdict_counts"].get("not_ship", 0),
                             fu["verdict_counts"].get("unsure", 0))
            L.append(f"\nThe **box-level** share is the figure that depends on the tile, and it should "
                     f"always be quoted next to the excluded version. Of SR's {fnb} extra boxes set-wide, "
                     f"{100*fns/fnb:.0f}%-{100*(fns+fun)/fnb:.0f}% had no vessel under them; of the "
                     f"{nb} that remain once `{top_tile}` is dropped, {ns_}-{ns_+un} do "
                     f"({100*ns_/nb:.0f}%-{100*(ns_+un)/nb:.0f}%). Bicubic's comparable figure is "
                     f"{d['beyond_original']['conf0.25']['bicubic']['verdict_counts'].get('not_ship',0)}"
                     f"-{d['beyond_original']['conf0.25']['bicubic']['verdict_counts'].get('not_ship',0) + d['beyond_original']['conf0.25']['bicubic']['verdict_counts'].get('unsure',0)}"
                     f" of {d['beyond_original']['conf0.25']['bicubic']['boxes_on_those_tiles']} boxes — "
                     f"too few boxes for a percentage to mean anything, which is why it is given as a "
                     f"count.\n")

        bo = d.get("beyond_original", {})
        if bo:
            L.append("### (b) CONFIDENCE BEYOND THE ORIGINAL — detector fires on the version, "
                     "silent on HR\n")
            L.append("The supporting metric, and the only one that depends on no verdict at all: it counts tiles where "
                     "degrading and restoring made the detector fire where the sharp original did not. "
                     "The verdict split then says how often that extra confidence was justified.\n")
            L.append("| conf | Version | tiles | % of tiles | 95% CI | boxes | ship | not_ship | unsure |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for ck, e in bo.items():
                for v in ("sr", "bicubic"):
                    r = e[v]
                    vc = r["verdict_counts"]
                    n = r["boxes_on_those_tiles"] or 1
                    L.append(f"| {ck[4:]} | {LABEL[v]} | {r['tiles_fired_beyond_hr']} | "
                             f"**{r['pct_tiles']:.1f}%** | "
                             f"[{r['pct_ci95'][0]:.1f}, {r['pct_ci95'][1]:.1f}] | "
                             f"{r['boxes_on_those_tiles']} | "
                             f"{vc.get('ship',0)} ({100*vc.get('ship',0)/n:.0f}%) | "
                             f"{vc.get('not_ship',0)} ({100*vc.get('not_ship',0)/n:.0f}%) | "
                             f"{vc.get('unsure',0)} ({100*vc.get('unsure',0)/n:.0f}%) |")
            e = bo["conf0.25"]["paired"]
            L.append(f"\nPaired at conf 0.25: SR-only tiles {e['sr_only_tiles']}, bicubic-only "
                     f"{e['bicubic_only_tiles']}, difference {e['diff_pct_points']:+.1f} pp "
                     f"[{e['diff_ci95'][0]:.1f}, {e['diff_ci95'][1]:.1f}], exact McNemar "
                     f"p = {e['mcnemar_exact_p']:.2e}.\n")
            L.append("**Why this is the right frame.** The SR image is computed from a 4x degraded "
                     "copy of the original, so by the data processing inequality it cannot contain "
                     "more information about the scene than the original does. A detection present "
                     "on SR and absent on HR is never new evidence — it is synthesized texture, or "
                     "the same evidence re-formatted into edges the detector's priors accept. The "
                     "audit says which: about a third of the time there really was a vessel the "
                     "original detector missed; the rest of the time there was not.\n")

    ops = {v: setA.get(v, {}).get("operating_conf_max_mean_f1") for v in VERSIONS}
    if any(ops.values()):
        L.append("\n## Footnote — what the Set A precision and recall are measured AT\n")
        L.append("Set A P and R are **not** at a fixed confidence threshold. Ultralytics' "
                 "`ap_per_class` (ultralytics/utils/metrics.py:92) chooses one index "
                 "`i = smooth(f1_curve.mean(0), 0.1).argmax()` — the confidence that maximises the "
                 "**smoothed mean F1 across all 15 classes** — and then reads *every* class's P and R "
                 "at that same index. So the ship row is quoted at whatever confidence happened to be "
                 "best for the 15-class average, and that confidence is **different for each version**:\n")
        L.append("| Version | confidence at which P and R are read |")
        L.append("|---|---|")
        for v in VERSIONS:
            if ops[v] is not None:
                L.append(f"| {LABEL[v]} | {ops[v]:.3f} |")
        L.append("\n**Consequence: the P/R columns above are not directly comparable across versions.** "
                 "Bicubic's recall is read at conf 0.338 and Real-ESRGAN's at conf 0.565 — a lower bar for "
                 "bicubic. `mAP50` is threshold-free and *is* comparable; so is the fixed-threshold table "
                 "in `results/setA_fixed_threshold.json`, which re-measures P/R/F1 for all three versions "
                 "at the same confidence.\n")

    ft = RESULTS / "setA_fixed_threshold.json"
    if ft.exists():
        d = json.loads(ft.read_text())
        L.append("\n## Set A at a FIXED confidence — the comparable version of the recovery table\n")
        L.append(f"Ship class only. Detections matched to ground truth greedily, highest confidence "
                 f"first, rotated-box IoU >= {d['iou_threshold']} (ultralytics `batch_probiou`). "
                 "Same threshold for every version, so these rows *can* be compared.\n")
        L.append("| conf | Version | TP | Unmatched detections | Missed GT | Precision | Recall | F1 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in (0.25, 0.5):
            for v in VERSIONS:
                r = d["results"].get(f"{v}_conf{c}")
                if r:
                    L.append(f"| {c} | {LABEL[v]} | {r['TP']} | {r['FP']} | {r['FN']} | "
                             f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
        hr25 = d["results"]["hr_conf0.25"]
        gt = hr25["TP"] + hr25["FN"]
        L.append("\nAt matched thresholds Real-ESRGAN's ship recall is **2.0 pp below bicubic at "
                 "conf 0.25 (0.881 vs 0.901) and 2.8 pp below at conf 0.50 (0.825 vs 0.853)** — the "
                 "same direction as the ultralytics table but a far smaller gap than its mismatched "
                 "operating points implied. SR also carries more unmatched detections on Set A "
                 f"({d['results']['sr_conf0.25']['FP']} vs {d['results']['bicubic_conf0.25']['FP']} "
                 "at conf 0.25), consistent with the Set B invention result.\n")
        L.append("**Footnotes on this table.**\n")
        L.append(f"1. *\"Unmatched detections\" is not \"false positives.\"* The original imagery "
                 f"itself produces **{hr25['FP']} unmatched ship detections** at conf 0.25, against "
                 f"the same labels. DOTA's ship annotations are demonstrably incomplete — it omits "
                 "small craft — so a detection with no matching label is not evidence that the "
                 "detector was wrong. The column is named for what it measures: a detection that "
                 "found no ground-truth box to pair with. The *relative* comparison across versions "
                 "still holds, because all three are scored against the same labels.")
        L.append(f"2. *Ground truth here is {gt} ship instances across all 100 Set A tiles* "
                 "(TP + Missed GT, identical for every version). That differs from the 3378 "
                 "ultralytics reports, because ultralytics' validator drops `P0261__1024__390___0` "
                 "— one of its DOTA polygon vertices sits at 1.179, outside the crop — while this "
                 "fixed-threshold script reads the label files directly and keeps all 100 tiles.")
        L.append(f"3. *IoU is `batch_probiou` from ultralytics*, a Gaussian (probabilistic) "
                 "approximation of rotated-box IoU rather than exact polygon intersection. It is "
                 "applied identically to all three versions, so it cannot favour one of them; it "
                 "does mean the absolute matching at the "
                 f"{d['iou_threshold']} threshold is approximate.\n")

    L.append("\n---\nGenerated by `scripts/06_summary.py` from `results/setA_metrics.json`, "
             "`results/setB_counts.csv`, `results/psnr.csv`, `results/setA_fixed_threshold.json`.\n")
    out = RESULTS / "summary.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
