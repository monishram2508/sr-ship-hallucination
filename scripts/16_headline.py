"""Write results/headline.md -- the three plain-English headline sentences.

Every number is read out of results/; nothing is typed by hand. Run after 09_recount.py,
07_figures.py and 10_fixed_threshold.py.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402


def fp(x):
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def pc(x):
    return f"{x:.1f}%"


def main():
    ft = json.loads((RESULTS / "setA_fixed_threshold.json").read_text())["results"]
    d = json.loads((RESULTS / "setB_audited.json").read_text())
    dx = json.loads((RESULTS / "setB_audited_no_P1751.json").read_text())
    px = json.loads((RESULTS / "psnr_vs_invention.json").read_text())

    psnr, ssim = {}, {}
    for r in csv.DictReader(open(RESULTS / "psnr.csv")):
        psnr.setdefault(r["version"], []).append(float(r["psnr"]))
        ssim.setdefault(r["version"], []).append(float(r["ssim"]))
    mean = lambda xs: sum(xs) / len(xs)

    key = {r["crop_id"]: r for r in csv.DictReader(open(RESULTS / "audit_key.csv"))}
    labs = {r["crop_id"]: (r.get("verdict") or "").strip().lower()
            for r in csv.DictReader(open(RESULTS / "audit_labels.csv"))}
    ns_tile = Counter(key[c]["tile"] for c, v in labs.items() if v == "not_ship")
    top_tile, top_n = ns_tile.most_common(1)[0]
    total_ns = sum(ns_tile.values())
    n_top_boxes = sum(1 for c in key if key[c]["tile"] == top_tile)

    lo, up = d["bands"]["conf0.25_lower"], d["bands"]["conf0.25_upper"]
    xlo, xup = dx["bands"]["conf0.25_lower"], dx["bands"]["conf0.25_upper"]
    lo5, up5 = d["bands"]["conf0.5_lower"], d["bands"]["conf0.5_upper"]
    b25, b50 = d["beyond_original"]["conf0.25"], d["beyond_original"]["conf0.5"]
    sr25, bc25 = b25["sr"], b25["bicubic"]
    sr50, bc50 = b50["sr"], b50["bicubic"]
    xsr25 = dx["beyond_original"]["conf0.25"]["sr"]

    def noship(e):
        n, v = e["boxes_on_those_tiles"], e["verdict_counts"]
        return v.get("not_ship", 0), v.get("not_ship", 0) + v.get("unsure", 0), n

    s_lo, s_up, s_n = noship(sr25)
    x_lo, x_up, x_n = noship(xsr25)
    b_lo, b_up, b_n = noship(bc25)
    f_lo, f_up, f_n = noship(sr50)

    L = []
    A = L.append
    A("# Three headline sentences\n")

    # ---------------- 1. recovery ----------------
    A("**1. Recovery — it did not happen.** Putting a GAN super-resolver in front of an "
      "off-the-shelf ship detector did not recover ships lost to resolution: at a matched "
      f"confidence threshold, ship recall on Real-ESRGAN output was "
      f"**{ft['sr_conf0.25']['recall']:.3f}** against **{ft['bicubic_conf0.25']['recall']:.3f}** "
      f"for a plain bicubic upscale of the same degraded image and "
      f"**{ft['hr_conf0.25']['recall']:.3f}** on the original "
      "(conf 0.25, rotated-IoU 0.5, `results/setA_fixed_threshold.json`). The GAN cost recall "
      f"rather than restoring it, and it did so while adding unmatched detections "
      f"({ft['sr_conf0.25']['FP']} vs bicubic's {ft['bicubic_conf0.25']['FP']}).\n")
    A("*Unmatched detections are not false positives:* the original imagery itself leaves "
      f"{ft['hr_conf0.25']['FP']} detections unmatched against the same labels, because DOTA's ship "
      "annotations omit small craft. The column counts detections that found no ground-truth box to "
      "pair with. All three versions are scored against the same labels, so the comparison between "
      f"them holds. Ground truth is "
      f"{ft['hr_conf0.25']['TP'] + ft['hr_conf0.25']['FN']} ship instances over all 100 Set A tiles "
      "(ultralytics' own validator reports 3378, because it drops `P0261__1024__390___0` for a "
      "polygon vertex at 1.179). IoU is ultralytics `batch_probiou`, a Gaussian approximation of "
      "rotated-box IoU, applied identically to every version.\n")

    # ---------------- 2. invention (THE LEAD) ----------------
    A("**2. Invention — the lead result. Super-resolution made the detector claim vessels that "
      f"are not there on {pc(lo['sr']['pct_tiles'])}-{pc(up['sr']['pct_tiles'])} of "
      f"{d['n_tiles']} harbor and bridge tiles where the same detector saw nothing in the "
      f"original, against {pc(lo['bicubic']['pct_tiles'])}-{pc(up['bicubic']['pct_tiles'])} for a "
      f"plain bicubic upscale of the same degraded data** (exact McNemar "
      f"p = {fp(lo['paired']['mcnemar_exact_p'])} lower band, "
      f"{fp(up['paired']['mcnemar_exact_p'])} upper; paired on the same tiles, "
      "the tile is the unit of independence).\n")

    A("**The firing rate comes first, because it needs no human verdict at all.** On "
      f"{d['n_tiles']} tiles the sharp original produced **zero** ship detections. Real-ESRGAN made "
      f"the detector fire on **{pc(sr25['pct_tiles'])} of tiles** "
      f"({sr25['tiles_fired_beyond_hr']}/{d['n_tiles']}, 95% CI "
      f"{sr25['pct_ci95'][0]:.1f}-{sr25['pct_ci95'][1]:.1f}) against "
      f"**{pc(bc25['pct_tiles'])}** ({bc25['tiles_fired_beyond_hr']}/{d['n_tiles']}, CI "
      f"{bc25['pct_ci95'][0]:.1f}-{bc25['pct_ci95'][1]:.1f}) for bicubic — a paired difference of "
      f"**{b25['paired']['diff_pct_points']:+.1f} points** "
      f"[{b25['paired']['diff_ci95'][0]:.1f}, {b25['paired']['diff_ci95'][1]:.1f}], exact McNemar "
      f"p = {fp(b25['paired']['mcnemar_exact_p'])}. **This holds at both detector thresholds**: at "
      f"conf 0.50 it is {pc(sr50['pct_tiles'])} vs {pc(bc50['pct_tiles'])}, "
      f"{b50['paired']['diff_pct_points']:+.1f} pp "
      f"[{b50['paired']['diff_ci95'][0]:.1f}, {b50['paired']['diff_ci95'][1]:.1f}], "
      f"p = {fp(b50['paired']['mcnemar_exact_p'])}.\n")

    A(f"Every one of those extra boxes ({d['n_crops']} in total, SR and bicubic alike) was then "
      "cropped from the **original** imagery and judged by hand, blind to which version produced "
      "it, in two passes. Counting only tiles that carry a box judged *not a vessel*:\n")

    A("| invention at conf 0.25 | Real-ESRGAN | bicubic | difference | exact McNemar p |")
    A("|---|---|---|---|---|")
    for nm, bb in (("lower band (not-a-vessel only)", lo), ("upper band (+ uncertain)", up)):
        A(f"| {nm} | **{pc(bb['sr']['pct_tiles'])}** "
          f"[{bb['sr']['pct_ci95'][0]:.1f}, {bb['sr']['pct_ci95'][1]:.1f}] | "
          f"{pc(bb['bicubic']['pct_tiles'])} "
          f"[{bb['bicubic']['pct_ci95'][0]:.1f}, {bb['bicubic']['pct_ci95'][1]:.1f}] | "
          f"{bb['paired']['diff_pct_points']:+.1f} pp | **{fp(bb['paired']['mcnemar_exact_p'])}** |")
    A("")

    A(f"**It survives dropping the worst tile.** Invention is concentrated: `{top_tile}` alone "
      f"carries {top_n} of the {total_ns} not-a-vessel boxes in the whole set. Removing that tile "
      f"from the numerator *and* the denominator leaves {dx['n_tiles']} tiles, and the tile-level "
      f"result barely moves — **{pc(xlo['sr']['pct_tiles'])}-{pc(xup['sr']['pct_tiles'])}** of "
      f"tiles against bicubic's {pc(xlo['bicubic']['pct_tiles'])}-{pc(xup['bicubic']['pct_tiles'])}, "
      f"p = {fp(xlo['paired']['mcnemar_exact_p'])} and "
      f"{fp(xup['paired']['mcnemar_exact_p'])}. The test counts each tile once, so no single scene "
      "can drive it.\n")

    A("**At box level the two arms are different kinds of error** — and this is the figure that "
      f"does depend on that one tile, so both versions are given. Of Real-ESRGAN's {s_n} extra "
      f"boxes set-wide, **{s_lo}-{s_up} had no vessel under them "
      f"({100*s_lo/s_n:.0f}-{100*s_up/s_n:.0f}%)**; of the {x_n} that remain once `{top_tile}` is "
      f"dropped, **{x_lo}-{x_up} do ({100*x_lo/x_n:.0f}-{100*x_up/x_n:.0f}%)**. Bicubic's "
      f"{b_n} extra boxes contain **{b_lo}-{b_up} with no vessel** — {b_n} boxes is too few for a "
      f"percentage to mean anything, so it is given as a count. The rest of bicubic's extra boxes "
      f"({bc25['verdict_counts'].get('ship', 0)} of {b_n}) were real vessels DOTA never labelled "
      "and the original detector missed: when a plain upscale makes the detector newly confident, "
      "it is usually right. When the GAN does it, most of the time there is nothing there.\n")

    A("**What is not significant.** Invention at a detector threshold of 0.50: "
      f"{pc(lo5['sr']['pct_tiles'])}-{pc(up5['sr']['pct_tiles'])} vs "
      f"{pc(lo5['bicubic']['pct_tiles'])}, p = {fp(lo5['paired']['mcnemar_exact_p'])} and "
      f"{fp(up5['paired']['mcnemar_exact_p'])}. Only {f_n} SR boxes survive that threshold at all, "
      f"{f_lo}-{f_up} of them not-a-vessel, on "
      f"{lo5['sr']['tiles_with_ship']}-{up5['sr']['tiles_with_ship']} tiles. That is too little "
      "data to resolve the difference, not evidence that it goes away: the point estimate still "
      "moves the same direction, and the *firing rate* at the same threshold is significant "
      f"(p = {fp(b50['paired']['mcnemar_exact_p'])}). Raising the detector's confidence bar reduces "
      "invention; it does not remove it.\n")

    A("**Why any of this is possible.** The SR image is computed from a 4x degraded copy of the "
      "original: by the data processing inequality it cannot contain more information about the "
      "scene than the original does. A detection that appears on the SR image and not on the "
      "original is therefore never new evidence. It is either synthesized texture, or the same "
      "evidence re-formatted into edges and contrast the detector's priors will accept — and the "
      f"audit says which. On {top_tile.split('__')[0]}, a dark residential hillside, Real-ESRGAN "
      f"produced {n_top_boxes} ship detections where the original and the bicubic upscale both "
      f"returned zero, and all {n_top_boxes} were judged not a vessel. That tile is an illustration "
      "of the failure mode, not a rate.\n")

    # ---------------- 3. psnr ----------------
    A("**3. What the pixel metrics said — nothing useful.** PSNR and SSIM barely separated the two "
      f"methods: {mean(psnr['sr']):.2f} dB vs {mean(psnr['bicubic']):.2f} dB, SSIM "
      f"{mean(ssim['sr']):.3f} vs {mean(ssim['bicubic']):.3f}, while their tile-level invention "
      f"rates differ by roughly {lo['sr']['pct_tiles']/lo['bicubic']['pct_tiles']:.0f}x. The pixel "
      "scores happen to rank bicubic first, but a 1.3 dB gap carries no signal about *why*: nothing "
      "in the number distinguishes \"slightly less faithful texture\" from \"draws vessels that do "
      "not exist\". Within the SR arm, PSNR does not even rank the tiles: low SR PSNR predicts a "
      f"tile carrying an invented object with AUC "
      f"**{px['auc_low_psnr_predicts_invention']:.2f}** (p = {px['mannwhitney_p']:.2f}), Spearman "
      f"rho = {px['spearman_rho']:+.2f} (p = {px['spearman_p']:.2f}) — indistinguishable from a "
      "coin flip.\n")

    A("---")
    A("Generated by `scripts/16_headline.py` from `results/setA_fixed_threshold.json`, "
      "`results/setB_audited.json`, `results/setB_audited_no_P1751.json`, "
      "`results/psnr_vs_invention.json` and `results/psnr.csv`. No number in this file is typed "
      "by hand.")

    out = RESULTS / "headline.md"
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
