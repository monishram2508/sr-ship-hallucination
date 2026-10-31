"""Phase 8: the write-up. Renders v2/results/summary.md and its two figures from the results files,
so every number in the summary is read from a file that code wrote, never typed by hand.

Reads v2/results/{stats.json, setA_metrics.json, setB_firing.json, setB_invention.json,
phase0_sr_audit.json, trainset_summary.json, checkpoint_runs.json, kaggle_prepare_report.json,
audit/build_summary.json, audit/audit_labels.csv} and v1's results/setB_audited.json. Writes
v2/results/summary.md, v2/results/figures/setB_firing_invention.png and
v2/results/figures/setA_recall_precision.png.

The verdict is prose, so check_claims() first tests every direction it asserts (whether each
seed-averaged 95% CI is below, above or across zero) and exits rather than write a verdict the
numbers no longer support.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, CONDITION_COLOURS, V1_RESULTS, V2_RESULTS  # noqa: E402

SEEDS = [str(s) for s in CFG["train"]["seeds"]]
DETECTORS = ["v1", "M_HR", "M_SR", "M_BIC"]           # fixed order, so each keeps its colour
VERSIONS = {"hr": "original", "bicubic": "bicubic", "sr": "Real-ESRGAN"}
C1, C2 = "C1_sr_images_MSR_vs_MHR", "C2_pipelines_MSR_on_sr_vs_MBIC_on_bicubic"
INV_LO, INV_HI = "setB_invention_lower_conf0.25_all", "setB_invention_upper_conf0.25_all"
RECALL, BEYOND = "setA_recall_conf0.25", "setB_firing_beyond_conf0.25_all"
ROWS = [  # (stats key, label, unit); the first three are the primary endpoints
    (INV_LO, "**Set B invention, unsure = real** (% of tiles)", "pp"),
    (INV_HI, "**Set B invention, unsure = invented** (% of tiles)", "pp"),
    (RECALL, "**Set A recall**", "rate"),
    ("setB_invention_lower_conf0.25_no_dominant", "invention, unsure = real, without P1751", "pp"),
    ("setB_invention_lower_conf0.5_all", "invention, unsure = real, conf 0.5", "pp"),
    ("setB_firing_raw_conf0.25_all", "Set B firing, raw", "pp"),
    (BEYOND, "Set B firing beyond the original", "pp"),
    ("setA_precision_conf0.25", "Set A precision", "rate"),
    ("setA_f1_conf0.25", "Set A F1", "rate"),
    ("setA_recall_conf0.5", "Set A recall, conf 0.5", "rate"),
]
CLAIMS = [  # what section 4 asserts about each seed-averaged 95% CI
    (C1, INV_LO, "below"), (C1, INV_HI, "below"), (C1, RECALL, "above"),
    (C2, INV_LO, "across"), (C2, INV_HI, "across"), (C2, RECALL, "below"),
]
STYLE = {"ink": "#2b2b2a", "muted": "#6b6a64", "grid": "#e4e3de", "surface": "#fcfcfb"}
POST = "added_after_first_results"         # stats.json block that is NOT pre-registered
EXCESS_ROWS = [("M_SR_on_sr_over_own_originals", "M_SR, Real-ESRGAN minus originals"),
               ("M_BIC_on_bicubic_over_own_originals", "M_BIC, bicubic minus originals"),
               ("M_HR_on_sr_over_own_originals", "M_HR, Real-ESRGAN minus originals"),
               ("v1_on_sr_over_own_originals", "v1 detector, Real-ESRGAN minus originals"),
               ("v1_on_bicubic_over_own_originals", "v1 detector, bicubic minus originals")]
# A visual reading of figures/mhr_false_ships_on_originals.png, written after looking at it.
GALLERY_CAPTION = (
    "By eye, they are mostly pier and pontoon ends (several with a moored boat beside the box rather "
    "than in it), bright rectangular roofs, faint rectangles on bare ground, a striped shoreline and one "
    "parked car. Two panels (second in the top row, third in the bottom row) show a white hull inside or "
    "across the box; those verdicts look wrong, which is the labelling noise of limitation 1 and the "
    "region rule of limitation 2. This caption is a visual reading, not a measurement.")


def load(rel, root=V2_RESULTS):
    return json.loads((root / rel).read_text())


def model_name(det, seed):
    return "v1_offshelf" if det == "v1" else f"{det}_s{seed}"


def cell(det, ver, get):
    """-> per-seed values of one (detector, version) cell (one value for v1), or None if missing."""
    vals = [get(model_name(det, s), ver) for s in ([""] if det == "v1" else SEEDS)]
    return None if any(v is None for v in vals) else vals


def invention(inv, band, conf="conf0.25"):
    def get(model, ver):
        e = inv.get(model, {}).get(ver)
        return e[conf][band]["all"]["pct"] if e else None
    return get


def firing(fire):
    return lambda model, ver: fire[model][ver]["conf0.25"]["raw"]["pct"]


def setA(sa, metric):
    return lambda model, ver: sa[model][ver]["ap50"] if metric == "ap50" else sa[model][ver]["conf0.25"][metric]


def fmt(x, unit):
    return f"{x:.1f}" if unit == "pp" else f"{x:.3f}"


def signed(x, unit):
    digits = 1 if unit == "pp" else 3
    return f"{round(x, digits) + 0.0:+.{digits}f}".replace("-", "−")     # + 0.0 turns -0.0 into 0.0


def mean_sd(vals, unit):
    if len(vals) == 1:
        return fmt(vals[0], unit)
    return f"{fmt(np.mean(vals), unit)} ± {fmt(np.std(vals, ddof=1), unit)}"


def ci_text(e, unit):
    lo, hi = e["seed_avg_diff_ci95"]
    return f"{signed(lo, unit)} to {signed(hi, unit)}"


def per_seed(e, unit):
    return ", ".join(signed(v["diff_pp" if unit == "pp" else "diff"], unit) for v in e["per_seed"].values())


def same_sign(e):
    return "1 run" if e["n_seeds"] == 1 else "yes" if e["same_sign_all_seeds"] else "no"


def zero(e):
    lo, hi = e["seed_avg_diff_ci95"]
    return "the CI excludes zero" if lo > 0 or hi < 0 else "the CI includes zero"


def direction(e, unit):
    if e["same_sign_all_seeds"]:
        return "consistent across all three seeds"
    return f"not consistent across seeds (per seed {per_seed(e, unit)})"


def check_claims(stats):
    bad = []
    for name, key, want in CLAIMS:
        lo, hi = stats["comparisons"][name][key]["seed_avg_diff_ci95"]
        got = "below" if hi < 0 else "above" if lo > 0 else "across"
        if got != want:
            bad.append(f"{name} {key}: CI {lo:+.3f}, {hi:+.3f} is {got} zero, the verdict says {want}")
    return bad


# ---------------------------------------------------------------------------------------- tables

def comparison_table(cmp, a, b):
    lines = [f"| Metric (conf 0.25 unless noted) | {a} | {b} | difference | 95% CI | per seed (0, 1, 2) | "
             "same sign in all seeds |", "|---|---|---|---|---|---|---|"]
    for key, label, unit in ROWS:
        e = cmp[key]
        lines.append(f"| {label} | {fmt(e['a_mean'], unit)} ± {fmt(e['a_sd'], unit)} | {fmt(e['b_mean'], unit)} ± "
                     f"{fmt(e['b_sd'], unit)} | {signed(e['diff_mean'], unit)} | {ci_text(e, unit)} | "
                     f"{per_seed(e, unit)} | {'yes' if e['same_sign_all_seeds'] else 'no'} |")
    ap_a = [cmp[RECALL]["per_seed"][s]["ap50_a"] for s in SEEDS]
    ap_b = [cmp[RECALL]["per_seed"][s]["ap50_b"] for s in SEEDS]
    d = [x - y for x, y in zip(ap_a, ap_b)]
    same = all(x > 0 for x in d) or all(x < 0 for x in d)
    lines.append(f"| Set A AP50 | {mean_sd(ap_a, 'rate')} | {mean_sd(ap_b, 'rate')} | {signed(np.mean(d), 'rate')} | "
                 f"not bootstrapped | {', '.join(signed(x, 'rate') for x in d)} | {'yes' if same else 'no'} |")
    return "\n".join(lines)


def mcnemar_table(stats):
    lines = ["| Comparison | unsure counted as | seed | A (% tiles) | B (% tiles) | tiles only A | tiles only B | "
             "exact McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for name, short in ((C1, "C1"), (C2, "C2")):
        for key, band in ((INV_LO, "real"), (INV_HI, "invented")):
            for s in SEEDS:
                e = stats["comparisons"][name][key]["per_seed"][s]
                lines.append(f"| {short} | {band} | {s} | {e['a_pct']:.1f} | {e['b_pct']:.1f} | {e['a_only_tiles']} | "
                             f"{e['b_only_tiles']} | {e['mcnemar_exact_p']:.3f} |")
    return "\n".join(lines)


def cells_table(sa, fire, inv):
    cols = [("invention, unsure = real (%)", invention(inv, "lower"), "pp"),
            ("invention, unsure = invented (%)", invention(inv, "upper"), "pp"),
            ("raw firing (%)", firing(fire), "pp"), ("recall", setA(sa, "recall"), "rate"),
            ("precision", setA(sa, "precision"), "rate"), ("AP50", setA(sa, "ap50"), "rate")]
    lines = ["| Detector | test images | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for det in DETECTORS:
        for ver, ver_label in VERSIONS.items():
            vals = [cell(det, ver, get) for _, get, _ in cols]
            text = ["not audited" if v is None else mean_sd(v, unit) for v, (_, _, unit) in zip(vals, cols)]
            name = "v1 detector (1 run)" if det == "v1" else det
            lines.append(f"| {name} | {ver_label} | " + " | ".join(text) + " |")
    return "\n".join(lines)


def excess_table(post):
    lines = ["| unsure counted as | detector, images | restored (% tiles) | originals (% tiles) | excess (points) | "
             "95% CI | per seed | same sign in all seeds |", "|---|---|---|---|---|---|---|---|"]
    for key, band in ((INV_LO, "real"), (INV_HI, "invented")):
        for name, label in EXCESS_ROWS:
            e = post["excess_over_own_originals"][name][key]
            lines.append(f"| {band} | {label} | {fmt(e['a_mean'], 'pp')} | {fmt(e['b_mean'], 'pp')} | "
                         f"{signed(e['diff_mean'], 'pp')} | {ci_text(e, 'pp')} | {per_seed(e, 'pp')} | {same_sign(e)} |")
        e = post["excess_difference_MSR_minus_MBIC"][key]
        lines.append(f"| {band} | **M_SR excess minus M_BIC excess** | | | {signed(e['diff_mean'], 'pp')} | "
                     f"{ci_text(e, 'pp')} | {per_seed(e, 'pp')} | {same_sign(e)} |")
    return "\n".join(lines)


def pairing_table(post):
    lines = ["| Comparison | endpoint | 9 seed pairings: A lower / A higher / tied | range of differences | "
             "pairings whose 95% CI excludes 0 | majority vote (2 of 3 seeds), A vs B | difference, 95% CI | "
             "McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for name, short in ((C1, "C1"), (C2, "C2")):
        for key, label, unit in ((INV_LO, "invention, unsure = real", "pp"),
                                 (INV_HI, "invention, unsure = invented", "pp"), (RECALL, "Set A recall", "rate")):
            e, m = post["cross_seed_pairs"][name][key], post["majority_vote"][name].get(key)
            ties = e["n_pairs"] - e["n_negative"] - e["n_positive"]
            maj = (f"{fmt(m['a_mean'], unit)} vs {fmt(m['b_mean'], unit)} | {signed(m['diff_mean'], unit)}, "
                   f"{ci_text(m, unit)} | {m['per_seed']['majority']['mcnemar_exact_p']:.3f}" if m
                   else "not applicable (tile yes/no only) | | ")
            lines.append(f"| {short} | {label} | {e['n_negative']} / {e['n_positive']} / {ties} | "
                         f"{signed(e['min'], unit)} to {signed(e['max'], unit)} | {e['n_ci_excluding_zero']} of "
                         f"{e['n_pairs']} | {maj} |")
    return "\n".join(lines)


def truncation_table(sa):
    capped, fixed = sa["v1_offshelf_v1settings"], sa["v1_offshelf"]
    lines = ["| v1 detector, conf 0.25 | recall as v1 ran it (max_det 300) | recall, uncapped | change |",
             "|---|---|---|---|"]
    for ver, label in VERSIONS.items():
        a, b = capped[ver]["conf0.25"]["recall"], fixed[ver]["conf0.25"]["recall"]
        lines.append(f"| {label} | {a:.3f} | {b:.3f} | {signed(b - a, 'rate')} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- figures

def offsets(n, width):
    return [(j - (n - 1) / 2) * width for j in range(n)]


def style_axis(ax, title, ylabel, ticks=None):
    ticks = ticks or [f"{v} images" for v in VERSIONS.values()]
    ax.set_facecolor(STYLE["surface"])
    ax.set_title(title, fontsize=10, color=STYLE["ink"], loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=STYLE["muted"])
    ax.set_xticks(range(len(ticks)), ticks)
    ax.tick_params(colors=STYLE["muted"], labelsize=8, length=0)
    ax.grid(axis="y", color=STYLE["grid"], lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def bar_panel(ax, values, title, ylabel):
    """Set B rates: bars from zero at the seed mean, one dot per seed, a note where unaudited."""
    width = 0.8 / len(DETECTORS)
    for dx, det in zip(offsets(len(DETECTORS), width), DETECTORS):
        for i, ver in enumerate(VERSIONS):
            vals = values[(det, ver)]
            if vals is None:
                ax.text(i + dx, 0.5, "not audited", rotation=90, ha="center", va="bottom", fontsize=6.5,
                        color=STYLE["muted"])
                continue
            ax.bar(i + dx, np.mean(vals), width * 0.88, color=CONDITION_COLOURS[det])
            if np.mean(vals) == 0:                     # a zero-height bar is invisible: say so
                ax.text(i + dx, 0.5, "0", ha="center", va="bottom", fontsize=7, color=STYLE["muted"])
            if len(vals) > 1:
                ax.scatter([i + dx] * len(vals), vals, s=9, color=STYLE["ink"], zorder=3, linewidths=0)
    style_axis(ax, title, ylabel)


def dot_panel(ax, values, title, ylabel):
    """Set A rates on a zoomed axis, so dots (one per seed) and a mean tick, never bars."""
    width = 0.8 / len(DETECTORS)
    for dx, det in zip(offsets(len(DETECTORS), width), DETECTORS):
        for i, ver in enumerate(VERSIONS):
            vals, x = values[(det, ver)], i + dx
            ax.hlines(np.mean(vals), x - width * 0.35, x + width * 0.35, color=CONDITION_COLOURS[det], lw=2)
            ax.scatter([x] * len(vals), vals, s=14, color=CONDITION_COLOURS[det], zorder=3,
                       edgecolors=STYLE["surface"], linewidths=0.8)
    style_axis(ax, title, ylabel)


EXCESS_GROUPS = [  # (tick label, [(detector, stats key)]), unsure counted as real
    ("Real-ESRGAN pipeline", [("v1", "v1_on_sr_over_own_originals"), ("M_HR", "M_HR_on_sr_over_own_originals"),
                              ("M_SR", "M_SR_on_sr_over_own_originals")]),
    ("bicubic pipeline", [("v1", "v1_on_bicubic_over_own_originals"),
                          ("M_BIC", "M_BIC_on_bicubic_over_own_originals")]),
]


def excess_panel(ax, post, title, ylabel):
    """Excess over the same detector on the originals: bar = seed mean, dots = seeds, line = 95% CI."""
    width = 0.8 / max(len(bars) for _, bars in EXCESS_GROUPS)
    for i, (_, bars) in enumerate(EXCESS_GROUPS):
        for dx, (det, key) in zip(offsets(len(bars), width), bars):
            e, x = post["excess_over_own_originals"][key][INV_LO], i + dx
            ax.bar(x, e["diff_mean"], width * 0.88, color=CONDITION_COLOURS[det])
            ax.vlines(x, *e["seed_avg_diff_ci95"], color=STYLE["ink"], lw=1.2, zorder=3)
            seeds = [v["diff_pp"] for v in e["per_seed"].values()]
            if len(seeds) > 1:
                ax.scatter([x] * len(seeds), seeds, s=9, color=STYLE["ink"], zorder=4, linewidths=0)
    ax.axhline(0, color=STYLE["muted"], lw=0.8)
    style_axis(ax, title, ylabel, [g for g, _ in EXCESS_GROUPS])


def figure(panels, out, detectors=DETECTORS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), facecolor=STYLE["surface"],
                             squeeze=False)
    for ax, (draw, values, title, ylabel) in zip(axes[0], panels):
        draw(ax, values, title, ylabel)
    labels = {"v1": "v1 detector (off-the-shelf, 1 run)", "M_HR": "M_HR (trained on originals)",
              "M_SR": "M_SR (originals + Real-ESRGAN)", "M_BIC": "M_BIC (originals + bicubic)"}
    fig.legend([Patch(color=CONDITION_COLOURS[d]) for d in detectors], [labels[d] for d in detectors],
               loc="lower center", ncol=2 if len(panels) == 1 else len(detectors), frameon=False, fontsize=8,
               labelcolor=STYLE["ink"])
    fig.tight_layout(rect=(0, 0.12 if len(panels) == 1 else 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=STYLE["surface"])
    plt.close(fig)


def make_figures(sa, fire, inv, post):
    grid = lambda get: {(d, v): cell(d, v, get) for d in DETECTORS for v in VERSIONS}  # noqa: E731
    fig_dir = V2_RESULTS / "figures"
    figure([(excess_panel, post, "Invention over the same detector on the originals (added after first results)",
             "points of tiles, unsure = real")], fig_dir / "setB_excess_over_own_originals.png", DETECTORS)
    figure([(bar_panel, grid(firing(fire)), "Set B: tiles with any ship detection (raw firing)", "% of 150 tiles"),
            (bar_panel, grid(invention(inv, "lower")), "Set B: tiles with an invented ship (unsure = real)",
             "% of 150 tiles")], fig_dir / "setB_firing_invention.png")
    figure([(dot_panel, grid(setA(sa, "recall")), "Set A recall", "recall, conf 0.25"),
            (dot_panel, grid(setA(sa, "precision")), "Set A precision", "precision, conf 0.25")],
           fig_dir / "setA_recall_precision.png")


# -------------------------------------------------------------------------------------- sections

def priority_verdicts(build):
    with open(V2_RESULTS / "audit" / "audit_labels.csv") as f:
        rows = list(csv.DictReader(f))
    first = [(r["verdict"] or "").strip() for r in rows[:build["n_priority_crops"]]]
    labelled = sum(1 for r in rows if (r["verdict"] or "").strip())
    return {v: first.count(v) for v in ("ship", "unsure", "not_ship")}, labelled


def queue_counts(name):
    with open(V2_RESULTS / "audit" / f"queue_{name}.csv") as f:
        verdicts = [(r["verdict"] or "").strip() for r in csv.DictReader(f)]
    return {"n": len(verdicts), **{v: verdicts.count(v) for v in ("ship", "unsure", "not_ship")}}


def recall_shift(sa):
    """v1's max_det truncation: -> (min, max) recall points lost, over versions and both thresholds."""
    d = [sa["v1_offshelf"][v][f"conf{c}"]["recall"] - sa["v1_offshelf_v1settings"][v][f"conf{c}"]["recall"]
         for v in VERSIONS for c in CFG["eval"]["confs"]]
    return 100 * min(d), 100 * max(d)


def floors(ref):
    """Invention (unsure = real) of each retrained condition on the original tiles."""
    return {c: ref[f"{c}_on_hr"][INV_LO]["mean"] for c in ("M_HR", "M_SR", "M_BIC")}


def excess(post, name, key=INV_LO):
    return post["excess_over_own_originals"][name][key]


def section_changes(stats, d):
    ref, ts, runs, build = stats["references"], d["trainset"], d["runs"]["runs"], d["build"]
    k = ts["by_split_kind"]
    n_pos, n_hard = k["train_pos"] + k["val_pos"], k["train_hard_neg"] + k["val_hard_neg"]
    n_neg = n_hard + k["train_rand_neg"] + k["val_rand_neg"]
    batches = [r["total_batches"] for r in runs]
    conds = CFG["train"]["conditions"]
    v1sr, v1bic = ref["v1_on_sr"], ref["v1_on_bicubic"]
    verdicts, labelled = priority_verdicts(build)
    n_gt = d["sa"]["v1_offshelf"]["hr"]["n_gt"]
    table = "\n".join(["| Condition | trained on | epochs |", "|---|---|---|"] + [
        f"| {c} | {' + '.join(VERSIONS[s] for s in conds[c]['sources'])} | {conds[c]['epochs']} |" for c in conds])
    return f"""## 1. What changed from v1, and why

v1 put Real-ESRGAN in front of an off-the-shelf DOTA detector (`yolov8s-obb`) that had never seen
super-resolved images. The detector reported ships on {v1sr['setB_firing_raw_conf0.25_all']['mean']:.1f}% of
the verified-empty Set B tiles after SR, against {v1bic['setB_firing_raw_conf0.25_all']['mean']:.1f}% after
bicubic upscaling, and the hand audit put invention at {v1sr[INV_LO]['mean']:.1f}–{v1sr[INV_HI]['mean']:.1f}%
against {v1bic[INV_LO]['mean']:.1f}–{v1bic[INV_HI]['mean']:.1f}%.

A reviewer made four points: the numbers looked worse than GAN super-resolution should produce; a
detector never trained on SR images mixes two effects, SR inventing detail and a detector meeting SR
artefacts for the first time; a small ship-only detector should be trained with and without SR
images and the experiment rerun; and v1 was a single run.

v2 responds point by point:

- **Phase 0** audits v1's SR pipeline for bugs (section 2).
- **Training set:** {ts['n_tiles']} tiles of 1024 px from {ts['n_sources']} DOTA *train* images
  ({n_pos} with ships, {n_neg} without, {n_hard} of those hard negatives), split 90/10 by source image.
- **Three detectors**, YOLOv8n-OBB started from COCO weights, ship class only, the same number of
  gradient steps ({min(batches):,}–{max(batches):,} batches of 8 per run), `last.pt` evaluated:

{table}

- **Three seeds** per condition, nine runs.
- **Same test sets as v1:** Set A ({n_gt:,} labelled ships on 100 tiles) and Set B (150 tiles with no
  ship labels and no v1 detection on the original).
- **Pre-registered comparisons** (`PLAN.md`, written before any v2 result): C1, M_SR vs M_HR on SR
  images (the reviewer's question); C2, M_SR on SR vs M_BIC on bicubic (each pipeline with a detector
  trained on its own kind of image, the headline); C3, references. Primary endpoints: Set B invention
  at conf 0.25 in both unsure bands, and Set A recall at conf 0.25.
- **A new blind audit.** {build['n_detections_pooled']:,} Set B detections at conf 0.25 were pooled into
  {build['n_clusters']:,} regions; {build['clusters_by_source']['v1_match']} inherited a v1 verdict, and
  {build['n_new_crops']:,} became new crops of the original tile, plus {build['n_repeat_crops']} hidden
  repeats of v1 crops. The first {build['n_priority_crops']} crops cover every cell the comparisons read;
  {labelled} of {build['n_crops_to_label']:,} crops are labelled. Of the {build['n_priority_crops']} priority crops,
  {verdicts['ship']} were judged a real vessel, {verdicts['unsure']} unsure and {verdicts['not_ship']} not a ship."""


def section_phase0(d):
    p0, sa, kg, fire = d["phase0"], d["sa"], d["kaggle"], d["fire"]
    refs = p0["reference"]
    exact = sum(1 for r in refs if r["official_pp0_vs_v1_cpu"] == float("inf"))
    mps = [r["v1_mps_vs_v1_cpu"] for r in refs]
    deg, grey = p0["degradation"], p0["greyscale"]
    col = grey["setB_firing_by_colour"]["sr_conf0.25"]
    lo, hi = recall_shift(sa)
    lowest = max(h["lowest_kept_conf"] for h in sa["_max_det_hits"])
    psnr = list(kg["check_sr_psnr_vs_local_db"].values())
    gap = {k: sa[k]["sr"]["conf0.25"]["recall"] - sa[k]["bicubic"]["conf0.25"]["recall"]
           for k in ("v1_offshelf_v1settings", "v1_offshelf")}
    v1fire = fire["v1_offshelf"]
    return f"""## 2. Phase 0: the v1 SR pipeline

- **No SR implementation bug.** The official `realesrgan` inference code (`pre_pad=0`) on the CPU is
  bit-identical to v1's code on the CPU on {exact} of {len(refs)} tiles; v1's Apple-GPU output differs from
  its CPU output only by rounding ({min(mps):.1f}–{max(mps):.1f} dB PSNR). The degraded and bicubic images
  recompute exactly for {deg['setA']['bic_recomputed_exact']}/{deg['setA']['n']} Set A and
  {deg['setB']['bic_recomputed_exact']}/{deg['setB']['n']} Set B tiles.
- **Greyscale tiles come out coloured.** {grey['setB']['n_grey']} of {grey['setB']['n_tiles']} Set B and
  {grey['setA']['n_grey']} of {grey['setA']['n_tiles']} Set A tiles are greyscale; Real-ESRGAN writes colour into
  every one of them, as the official code does for 3-channel input. v1's SR firing was mostly on colour
  tiles ({col['colour_fired']}/{col['n_colour']} against {col['grey_fired']}/{col['n_grey']} grey), and turning the SR
  output back to grey moves the v1 detector's SR firing from {v1fire['sr']['conf0.25']['raw']['pct']:.1f}% to
  {v1fire['sr_greyfix']['conf0.25']['raw']['pct']:.1f}%.
- **A real v1 bug, in the evaluation.** v1 called the detector with ultralytics' default `max_det=300`,
  and two dense harbour tiles hit that cap. That understated v1's Set A recall by {lo:.1f}–{hi:.1f}
  points for every version at both thresholds; at conf 0.25 the SR-minus-bicubic recall gap moves from
  {signed(gap['v1_offshelf_v1settings'], 'rate')} to {signed(gap['v1_offshelf'], 'rate')}. v1's public README still shows the capped numbers; the corrected ones are
  here. v2 predicts with `max_det=1000`, and every image that reached it kept boxes down to conf
  {lowest:.3f} or lower, so no count above 0.25 is affected.

{truncation_table(sa)}

- **Training images match the test pipeline across devices.** The bicubic and SR training images were
  made on the training GPU from the uploaded originals. A later run of the same notebook on the same
  image and GPU type found the original and degraded inputs bit-identical to the local ones for
  {kg['n_tiles'] - max(kg['hr_hash_mismatch'], kg['lr_hash_mismatch'])} of {kg['n_tiles']} tiles, and its SR output within
  {min(psnr):.1f}–{max(psnr):.1f} dB PSNR of local SR on {len(psnr)} check tiles (rounding level)."""


def section_results(stats, d):
    cmp = stats["comparisons"]
    return f"""## 3. Results

Every number is per seed, then mean ± SD over the three seeds (the v1 detector is one run). The
95% CI is a tile bootstrap of the seed-averaged difference (10,000 resamples, seed 0): it carries
tile-to-tile variation, the SD carries seed-to-seed variation. Primary endpoints are in bold. Set B
rates are the share of 150 tiles with at least one detection (firing) or at least one invented ship
(invention); "unsure = real" and "unsure = invented" are v1's two treatments of `unsure` verdicts.

### C1: on SR images, trained with SR (M_SR, A) vs never saw SR (M_HR, B)

{comparison_table(cmp[C1], "M_SR on SR", "M_HR on SR")}

### C2: each pipeline with its own detector, M_SR on SR (A) vs M_BIC on bicubic (B), the headline

{comparison_table(cmp[C2], "M_SR on SR", "M_BIC on bicubic")}

#### C2, added after first results: invention over each pipeline's own detector on the originals

C2 compares two different detectors, each of which also invents on the untouched originals. This
table, added after the first results, subtracts that floor: invention on the restored images minus
the same detector on the original tiles, paired by tile and seed. The 56 crops it needed were
labelled after the first results (limitation 1).

{excess_table(stats[POST])}

![Invention over the same detector on the originals](figures/setB_excess_over_own_originals.png)

Bars are seed means, dots the seeds, lines the 95% CIs (unsure = real).

### Per-seed tile tests for the primary invention endpoints

{mcnemar_table(stats)}

### Seed-pairing-free check, added after first results

C1 and C2 pair seed 0 with seed 0, and so on, but seeds of different conditions share nothing. These
checks drop the pairing: every one of the 9 (A seed, B seed) combinations, and a majority vote in which
a tile counts as invented when at least 2 of a condition's 3 seeds invent on it.

{pairing_table(stats[POST])}

### C3: references, and every cell

"Not audited" marks a secondary cell that still has unlabelled crops; it is left out whole, never
counted partially. M_HR on the original images is the v2 ceiling.

{cells_table(d['sa'], d['fire'], d['inv'])}

![Set B firing and invention](figures/setB_firing_invention.png)

![Set A recall and precision](figures/setA_recall_precision.png)

Bars and ticks are seed means; dots are the individual seeds.

#### What a retrained detector calls a ship on the originals, added after first results

![M_HR detections on original Set B tiles audited not_ship](figures/mhr_false_ships_on_originals.png)

The {d['gallery']['n']} highest-confidence M_HR detections on the original Set B tiles that the audit judged
`not_ship`, one per audited region, from {d['gallery']['n_tiles']} tiles (confidence
{d['gallery']['conf_range'][0]:.2f}–{d['gallery']['conf_range'][1]:.2f}), each cropped from the original tile with
the box in magenta. {GALLERY_CAPTION}"""


def section_verdict(stats, d):
    c1, c2, ref = stats["comparisons"][C1], stats["comparisons"][C2], stats["references"]
    v1sr, post, fl = ref["v1_on_sr"], stats[POST], floors(ref)
    msr, mbic = excess(post, "M_SR_on_sr_over_own_originals"), excess(post, "M_BIC_on_bicubic_over_own_originals")
    mhr, v1s = excess(post, "M_HR_on_sr_over_own_originals"), excess(post, "v1_on_sr_over_own_originals")
    v1b, dd = excess(post, "v1_on_bicubic_over_own_originals"), post["excess_difference_MSR_minus_MBIC"][INV_LO]
    tie = ("" if not (msr["diff_mean"] > mbic["diff_mean"] and fl["M_SR"] < fl["M_BIC"]) else
           f" C2's raw rates tie although SR's excess is the larger because M_SR also has the lower rate on "
           f"the originals ({fl['M_SR']:.1f}% against {fl['M_BIC']:.1f}%).")
    pairs = {n: post["cross_seed_pairs"][n][INV_LO] for n in (C1, C2)}
    votes = {n: post["majority_vote"][n][INV_LO] for n in (C1, C2)}
    raw_hr = [ref[f"{c}_on_hr"]["setB_firing_raw_conf0.25_all"]["mean"] for c in ("M_HR", "M_SR", "M_BIC")]
    return f"""## 4. Verdict

**Q1. Does training on SR images help on SR images? Yes.** On SR images, M_SR invented ships on
{c1[INV_LO]['a_mean']:.1f}–{c1[INV_HI]['a_mean']:.1f}% of Set B tiles against {c1[INV_LO]['b_mean']:.1f}–{c1[INV_HI]['b_mean']:.1f}%
for M_HR, {abs(c1[INV_LO]['diff_mean']):.1f}–{abs(c1[INV_HI]['diff_mean']):.1f} points fewer (95% CIs {ci_text(c1[INV_LO], 'pp')} and
{ci_text(c1[INV_HI], 'pp')}). With unsure counted as invented the drop is {direction(c1[INV_HI], 'pp')}; with
unsure counted as real it is {direction(c1[INV_LO], 'pp')}. Recall on SR images rose from
{c1[RECALL]['b_mean']:.3f} to {c1[RECALL]['a_mean']:.3f} ({ci_text(c1[RECALL], 'rate')}), {direction(c1[RECALL], 'rate')}.

**Q2. With each detector trained on its own kind of image, does the SR pipeline still invent more
than the bicubic one? No.** M_SR on SR invented on {c2[INV_LO]['a_mean']:.1f}% of tiles and M_BIC on bicubic on
{c2[INV_LO]['b_mean']:.1f}% (unsure = real), {c2[INV_HI]['a_mean']:.1f}% and {c2[INV_HI]['b_mean']:.1f}% (unsure = invented): differences
of {signed(c2[INV_LO]['diff_mean'], 'pp')} and {signed(c2[INV_HI]['diff_mean'], 'pp')} points, 95% CIs {ci_text(c2[INV_LO], 'pp')} and
{ci_text(c2[INV_HI], 'pp')}; per seed the differences are {per_seed(c2[INV_LO], 'pp')} and
{per_seed(c2[INV_HI], 'pp')}. The bicubic pipeline kept slightly higher recall, {c2[RECALL]['b_mean']:.3f} against {c2[RECALL]['a_mean']:.3f}
({ci_text(c2[RECALL], 'rate')}), {direction(c2[RECALL], 'rate')}.
v1's gap, {v1sr[INV_LO]['mean']:.1f}–{v1sr[INV_HI]['mean']:.1f}% against {ref['v1_on_bicubic'][INV_LO]['mean']:.1f}–{ref['v1_on_bicubic'][INV_HI]['mean']:.1f}%, does
not survive matched training.

**What that does and does not mean.**

- **The false ships did not go away.** The retrained detectors invent on {fl['M_HR']:.1f}% (M_HR),
  {fl['M_SR']:.1f}% (M_SR) and {fl['M_BIC']:.1f}% (M_BIC) of the *original* Set B tiles (unsure = real), where v1's
  detector found nothing by construction.
- **Over each detector's own originals (added after first results).** Invention on the restored
  images minus the same detector on the originals is {signed(msr['diff_mean'], 'pp')} points of tiles for M_SR on SR
  ({ci_text(msr, 'pp')}; {zero(msr)}) and {signed(mbic['diff_mean'], 'pp')} for M_BIC on bicubic ({ci_text(mbic, 'pp')}; {zero(mbic)}).
  For v1's detector the same excesses are {signed(v1s['diff_mean'], 'pp')} and {signed(v1b['diff_mean'], 'pp')}. The difference
  between the two v2 excesses is {signed(dd['diff_mean'], 'pp')} points ({ci_text(dd, 'pp')}; {zero(dd)}; per seed
  {per_seed(dd, 'pp')}).{tie}
- **The reviewer's second point holds inside v2.** M_HR, which never saw an SR image, has an excess of
  {signed(mhr['diff_mean'], 'pp')} points on SR images over its own originals ({ci_text(mhr, 'pp')}), close to the v1
  detector's {signed(v1s['diff_mean'], 'pp')}; M_SR, trained with SR images, has {signed(msr['diff_mean'], 'pp')} (both added after
  first results).
- **Seed pairing does not drive C1 or C2 (added after first results).** Over all 9 seed pairings, M_SR
  invents less than M_HR on SR images in {pairs[C1]['n_negative']} and more in {pairs[C1]['n_positive']}, while C2 splits
  {pairs[C2]['n_negative']} lower, {pairs[C2]['n_positive']} higher and {pairs[C2]['n_pairs'] - pairs[C2]['n_negative'] - pairs[C2]['n_positive']} tied. A 2-of-3 majority vote gives
  {signed(votes[C1]['diff_mean'], 'pp')} points for C1 ({ci_text(votes[C1], 'pp')}) and {signed(votes[C2]['diff_mean'], 'pp')} for C2
  ({ci_text(votes[C2], 'pp')}).
- **What C2 rules out.** At 150 tiles and three seeds, an SR pipeline inventing on more than about
  {c2[INV_LO]['seed_avg_diff_ci95'][1]:.1f} points of tiles above the bicubic pipeline is unlikely; a smaller gap cannot be
  excluded.
- **Firing is not invention.** The v2 detectors fire on {min(raw_hr):.1f}–{max(raw_hr):.1f}% of the original
  Set B tiles (condition means). Firing beyond the original, a secondary measure, is
  {c2[BEYOND]['a_mean']:.1f}% for M_SR on SR against {c2[BEYOND]['b_mean']:.1f}% for M_BIC on bicubic
  ({signed(c2[BEYOND]['diff_mean'], 'pp')} points, 95% CI {ci_text(c2[BEYOND], 'pp')}; per seed
  {per_seed(c2[BEYOND], 'pp')}), yet audited invention is the same for both pipelines (Q2)."""


def kappa_disagreements(inv):
    pairs = inv["_kappa"]["pairs"]
    diff = [p for p in pairs if p["v2"] != p["v1_pass1"]]
    return len(diff), sum(1 for p in diff if p["v2"] == "not_ship")


def section_limitations(d):
    k, inv, queue = d["inv"]["_kappa"], d["inv"], d["queue"]
    n_diff, n_harsher = kappa_disagreements(inv)
    real_v1 = d["v1_audit"]["verdict_tally"]["sr_ship"]
    return f"""## 5. Limitations

1. **One labeller.** Intra-rater agreement on {k['n_repeats']} hidden repeats: Cohen's κ
   {k['vs_v1_pass1']:.2f} against the v1 first-pass verdicts on the same crops ({100 * k['agreement_pass1']:.0f}%
   raw agreement) and {k['vs_v1_final']:.2f} against the v1 final verdicts. {n_harsher} of the {n_diff}
   disagreements are a v2 `not_ship` where v1 said `unsure` or `ship`, so v2's invention rates may run
   slightly high. The labelling was blind to detector and image version, so this should not favour a
   condition. The {queue['n']} crops behind the excess analysis were labelled after the first results, from a
   queue holding only regions that M_SR or M_BIC drew on the original tiles ({queue['ship']} ship,
   {queue['unsure']} unsure, {queue['not_ship']} not_ship), so that batch was not blind to the cell it came from.
2. **One verdict per region.** A region's verdict is judged on its seed box and applies to every
   detection within probIoU 0.5 of it, so a member box can sit slightly differently from the box that was
   judged (the second panel of the gallery in section 3 is an example).
3. **One detector family and one SR model.** YOLOv8n-OBB in v2 (YOLOv8s-OBB in v1), and Real-ESRGAN
   x4plus, a GAN trained on everyday photos, not on satellite imagery.
4. **DOTA, not a target sensor.** DOTA mixes aerial and satellite imagery at varying ground resolution,
   and the degradation is synthetic (Gaussian blur σ = {CFG['blur_sigma']:g}, then {CFG['scale']}× area downsampling).
5. **Incomplete labels.** DOTA omits small craft: v1's audit found {real_v1} real vessels among the boxes
   its detector drew on SR Set B tiles. Training negatives may hold unlabelled vessels too; that affects
   every condition alike.
6. **Set B is empty only for v1's detector.** The v2 detectors fire on the original tiles, which is why
   v2 reports audited invention rather than firing.
7. **Not every cell is audited.** {len(inv['_not_audited'])} secondary (detector, version) cells still have
   unlabelled crops and are left out whole. The labelling order (priority cells first) was decided after
   the audit was built and before any verdict existed; the originals queue (limitation 1) came later.
8. **Three seeds**, and GPU training is not bit-exact (mixed precision, cuDNN); seed spread is shown
   everywhere rather than assumed away."""


def caveat(stats):
    fl, post = floors(stats["references"]), stats[POST]
    sr, bic = excess(post, "M_SR_on_sr_over_own_originals"), excess(post, "M_BIC_on_bicubic_over_own_originals")
    v1s, v1b = excess(post, "v1_on_sr_over_own_originals"), excess(post, "v1_on_bicubic_over_own_originals")
    return (f"these small detectors also invent on {min(fl.values()):.1f}–{max(fl.values()):.1f}% of the untouched "
            f"originals; over that floor (an analysis added after first results), SR adds "
            f"{sr['diff_mean']:.1f} points and bicubic {bic['diff_mean']:.1f}, against {v1s['diff_mean']:.1f} and "
            f"{v1b['diff_mean']:.1f} for v1's detector.")


def tldr(stats, d):
    c1, c2 = stats["comparisons"][C1], stats["comparisons"][C2]
    lo, hi = recall_shift(d["sa"])
    return f"""**TL;DR**

- **Q1, does training on SR help on SR images? Yes.** Invented ships on SR images fall from {c1[INV_LO]['b_mean']:.1f}% to {c1[INV_LO]['a_mean']:.1f}% of verified-empty tiles, and recall rises from {c1[RECALL]['b_mean']:.3f} to {c1[RECALL]['a_mean']:.3f}.
- **Q2, under matched training does the SR pipeline still invent more than bicubic? No detectable gap:** {c2[INV_LO]['a_mean']:.1f}% against {c2[INV_LO]['b_mean']:.1f}% of tiles (95% CI {ci_text(c2[INV_LO], 'pp')} points); bicubic keeps slightly higher recall ({c2[RECALL]['b_mean']:.3f} against {c2[RECALL]['a_mean']:.3f}).
- **Caveat:** {caveat(stats)}
- **v1 correction:** a `max_det=300` cap understated v1's Set A recall by {lo:.1f}–{hi:.1f} points; Phase 0 found no SR bug."""


def reviewer_summary(stats, d):
    c1, c2 = stats["comparisons"][C1], stats["comparisons"][C2]
    lo, hi = recall_shift(d["sa"])
    ci_lo, ci_hi = c2[INV_LO]["seed_avg_diff_ci95"]
    return (f"I retrained, as you suggested. A small ship-only detector (YOLOv8n-OBB, {d['trainset']['n_tiles']} DOTA "
            "tiles) was trained three ways, on originals only, plus Real-ESRGAN versions, or plus bicubic versions, "
            "three seeds each, then tested on v1's tiles with a new blind audit.\n\n"
            f"On SR images, training with SR cut invented ships from {c1[INV_LO]['b_mean']:.1f}% to "
            f"{c1[INV_LO]['a_mean']:.1f}% of verified-empty tiles and raised recall from {c1[RECALL]['b_mean']:.3f} to "
            f"{c1[RECALL]['a_mean']:.3f}. With each detector trained on its own kind of image, the SR and bicubic "
            f"pipelines invented equally often ({c2[INV_LO]['a_mean']:.1f}% vs {c2[INV_LO]['b_mean']:.1f}%, 95% CI "
            f"{signed(ci_lo, 'pp')} to {signed(ci_hi, 'pp')} points), so v1's SR-versus-bicubic gap does not survive; "
            f"bicubic kept slightly higher recall ({c2[RECALL]['b_mean']:.3f} vs {c2[RECALL]['a_mean']:.3f}). Caveat: "
            f"{caveat(stats)} Phase 0 found no SR bug, but v1's max_det=300 cap understated its Set A recall by "
            f"{lo:.1f}–{hi:.1f} points.")


def main():
    d = {"sa": load("setA_metrics.json"), "fire": load("setB_firing.json"), "inv": load("setB_invention.json"),
         "phase0": load("phase0_sr_audit.json"), "trainset": load("trainset_summary.json"),
         "runs": load("checkpoint_runs.json"), "kaggle": load("kaggle_prepare_report.json"),
         "build": load("audit/build_summary.json"), "v1_audit": load("setB_audited.json", V1_RESULTS),
         "gallery": load("mhr_false_ships_on_originals.json"), "queue": queue_counts("hr_floor")}
    stats = load("stats.json")
    gates = {f: d[k]["_regression_gate"]["passed"] for f, k in (("setA_metrics", "sa"), ("setB_invention", "inv"))}
    if not all(gates.values()):
        sys.exit(f"a v1 regression gate failed, nothing downstream is trusted: {gates}")
    bad = check_claims(stats)
    if bad:
        sys.exit("the verdict in section 4 no longer matches the numbers; rewrite it first:\n  " + "\n  ".join(bad))
    make_figures(d["sa"], d["fire"], d["inv"], stats[POST])
    email = reviewer_summary(stats, d)
    parts = ["# v2 results: does training the detector on super-resolved images remove the invented ships?\n\n"
             "Generated by `v2/scripts/11_summary.py` from the files in `v2/results/`; do not edit by hand. "
             "Everything marked *added after first results* was decided after the pre-registered numbers were "
             "seen and is not a pre-registered endpoint.", tldr(stats, d),
             section_changes(stats, d), section_phase0(d), section_results(stats, d), section_verdict(stats, d),
             section_limitations(d), f"## 6. Summary for the reviewer ({len(email.split())} words)\n\n{email}"]
    out = V2_RESULTS / "summary.md"
    out.write_text("\n\n".join(parts) + "\n")
    print(f"reviewer summary: {len(email.split())} words\n-> {out}\n-> {V2_RESULTS / 'figures'}")


if __name__ == "__main__":
    main()
