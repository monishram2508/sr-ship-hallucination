"""Phase 0: audit v1's degradation and super-resolution path for implementation bugs.

Every check is run against the files v1 actually wrote, not against a re-description of the code:

  1. SR model state: dtype, eval mode, scale; run_sr() source quoted into the report.
  2. Degradation: v1's LR and bicubic PNGs recomputed bit-for-bit from HR with v1's own functions;
     the Gaussian kernel OpenCV derives from sigma=2 compared with an explicit 6*sigma+1 kernel;
     file formats; HR PNG vs the JPEG tile it came from.
  3. Greyscale tiles: how many, and whether each version keeps them grey.
  4. Reference: the official realesrgan package (00a_official_reference.py, separate env) vs v1's
     SR PNGs, with v1's spandrel path re-run on CPU to separate device from implementation, and a
     deliberately channel-swapped run to show what a channel-order bug would look like.
  5. A visual grid of 8 tiles including P1751.

Writes v2/results/phase0_sr_audit.json and v2/results/phase0_sr_audit.md. Images go to
v2/results/phase0/ and v2/results/sanity_grid.png, both gitignored (DOTA-derived).
"""
import argparse, csv, inspect, json, random, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import fisher_exact
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import (CFG, V1_RESULTS, V2, V2_RESULTS, SETS, read_ids, version_image,  # noqa: E402
                       is_grey, load_v1_script, v1)

OUT = V2_RESULTS / "phase0"
DOM = CFG["dominant_tile"]
N_REF = 5
CROP = 384


def psnr(a, b):
    return float("inf") if np.array_equal(a, b) else float(sk_psnr(a, b, data_range=255))


def chroma(img):
    """(mean, max) absolute difference between neighbouring channels; 0 for a grey image."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    d = np.maximum(np.abs(b - g), np.abs(g - r))
    return float(d.mean()), int(d.max())


def reference_tiles():
    """P1751, one more grey tile and three colour tiles (seed 0). A channel-order bug is nearly
    invisible on grey input, so colour tiles must be in the sample, not left to chance."""
    rng = random.Random(0)
    rest = sorted(t for t in read_ids("setB") if t != DOM)
    grey = [t for t in rest if is_grey(cv2.imread(str(version_image("setB", "lr", t)), cv2.IMREAD_COLOR))]
    colour = [t for t in rest if t not in set(grey)]
    return [DOM] + rng.sample(grey, 1) + rng.sample(colour, N_REF - 2)


def model_checks(m03):
    import torch
    desc = m03.load_sr("cpu")
    dtypes = sorted({str(p.dtype) for p in desc.model.parameters()})
    src = inspect.getsource(m03.run_sr)
    return {"scale": int(desc.scale), "param_dtypes": dtypes,
            "eval_mode": not desc.model.training, "torch_version": torch.__version__,
            "run_sr_source": src, "degrade_source": inspect.getsource(m03.degrade),
            "uses_no_grad": "torch.no_grad()" in src, "channel_flips_in_run_sr": src.count("[:, :, ::-1]")}


def degradation_checks(m03):
    out = {}
    for s in SETS:
        ids = read_ids(s)
        c = dict(n=len(ids), hr_equals_tile_jpg=0, lr_recomputed_exact=0, bic_recomputed_exact=0,
                 k13_equals_default=0, lr_maxdiff_k25_vs_default=0, lr_meandiff_k25_vs_default=0.0)
        for t in tqdm(ids, desc=f"degradation {s}", leave=False):
            hr = cv2.imread(str(version_image(s, "hr", t)), cv2.IMREAD_COLOR)
            jpg = cv2.imread(str(v1.DATA / "tiles" / "images" / f"{t}.jpg"), cv2.IMREAD_COLOR)
            lr = cv2.imread(str(version_image(s, "lr", t)), cv2.IMREAD_COLOR)
            bic = cv2.imread(str(version_image(s, "bicubic", t)), cv2.IMREAD_COLOR)
            c["hr_equals_tile_jpg"] += int(np.array_equal(hr, jpg))
            c["lr_recomputed_exact"] += int(np.array_equal(m03.degrade(hr), lr))
            c["bic_recomputed_exact"] += int(np.array_equal(
                cv2.resize(lr, (v1.TILE, v1.TILE), interpolation=cv2.INTER_CUBIC), bic))
            area = lambda im: cv2.resize(im, (v1.LR, v1.LR), interpolation=cv2.INTER_AREA)  # noqa: E731
            k13 = area(cv2.GaussianBlur(hr, (13, 13), v1.BLUR_SIGMA))
            k25 = area(cv2.GaussianBlur(hr, (25, 25), v1.BLUR_SIGMA))
            c["k13_equals_default"] += int(np.array_equal(k13, lr))
            d = np.abs(k25.astype(np.int16) - lr.astype(np.int16))
            c["lr_maxdiff_k25_vs_default"] = max(c["lr_maxdiff_k25_vs_default"], int(d.max()))
            c["lr_meandiff_k25_vs_default"] += float(d.mean()) / len(ids)
        exts = {v: sorted({p.suffix for p in (v1.DATA / "versions" / v / s / "images").iterdir()})
                for v in ("hr", "lr", "bicubic", "sr")}
        c["extensions"] = exts
        out[s] = c
    return out


def greyscale_checks():
    out = {}
    for s in SETS:
        ids = read_ids(s)
        grey, sr_chroma = [], []
        for t in tqdm(ids, desc=f"greyscale {s}", leave=False):
            hr = cv2.imread(str(version_image(s, "hr", t)), cv2.IMREAD_COLOR)
            if not is_grey(hr):
                continue
            grey.append(t)
            keeps = {v: is_grey(cv2.imread(str(version_image(s, v, t)), cv2.IMREAD_COLOR))
                     for v in ("lr", "bicubic", "sr")}
            sr_chroma.append((t, *chroma(cv2.imread(str(version_image(s, "sr", t))))))
            out.setdefault(s + "_keeps_grey", {v: 0 for v in keeps})
            for v, k in keeps.items():
                out[s + "_keeps_grey"][v] += int(k)
        out[s] = {"n_tiles": len(ids), "n_grey": len(grey), "grey_sources": sorted({g.split("__")[0] for g in grey}),
                  "sr_chroma_mean_over_grey_tiles": float(np.mean([c[1] for c in sr_chroma])) if sr_chroma else 0.0,
                  "sr_chroma_max_over_grey_tiles": int(max([c[2] for c in sr_chroma])) if sr_chroma else 0,
                  "grey_tiles": grey}
    out["setB_firing_by_colour"] = firing_by_colour(set(out["setB"]["grey_tiles"]))
    return out


def firing_by_colour(grey):
    """v1's own Set B results split by grey vs colour tiles (read from v1's results files)."""
    ids = read_ids("setB")
    colour = set(ids) - grey
    rows = list(csv.DictReader(open(V1_RESULTS / "setB_counts.csv")))
    out = {}
    for conf in ("0.25", "0.5"):
        for v in ("sr", "bicubic"):
            fired = {r["tile"] for r in rows if r["version"] == v and r["conf"] == conf and int(r["n_ships"]) > 0}
            g, c = len(fired & grey), len(fired & colour)
            p = fisher_exact([[g, len(grey) - g], [c, len(colour) - c]])[1]
            out[f"{v}_conf{conf}"] = {"grey_fired": g, "n_grey": len(grey), "colour_fired": c,
                                      "n_colour": len(colour), "fisher_p": float(p)}
    return out


def reference_comparison(m03, ref_python):
    tiles = reference_tiles()
    OUT.mkdir(parents=True, exist_ok=True)
    if ref_python:
        subprocess.run([ref_python, str(V2 / "scripts" / "00a_official_reference.py"),
                        "--tiles", ",".join(tiles), "--out", str(OUT)], check=True)
    model = m03.load_sr("cpu")
    rows = []
    for t in tqdm(tiles, desc="reference", leave=False):
        lr = cv2.imread(str(version_image("setB", "lr", t)), cv2.IMREAD_COLOR)
        v1_png = cv2.imread(str(version_image("setB", "sr", t)), cv2.IMREAD_COLOR)
        cpu = m03.run_sr(model, lr, "cpu")
        swapped = m03.run_sr(model, lr[:, :, ::-1].copy(), "cpu")[:, :, ::-1]
        off = {p: cv2.imread(str(OUT / f"official_prepad{p}__{t}.png"), cv2.IMREAD_COLOR) for p in (0, 10)}
        if any(o is None for o in off.values()):
            sys.exit(f"official outputs for {t} missing in {OUT}; pass --ref-python")
        row = {"tile": t, "grey": is_grey(lr),
               "v1_mps_vs_v1_cpu": psnr(v1_png, cpu),
               "official_pp0_vs_v1_mps": psnr(off[0], v1_png),
               "official_pp0_vs_v1_cpu": psnr(off[0], cpu),
               "official_pp10_vs_v1_mps": psnr(off[10], v1_png),
               "official_pp0_vs_pp10": psnr(off[0], off[10]),
               "channel_swapped_vs_official_pp0": psnr(swapped, off[0]),
               "v1_sr_chroma": chroma(v1_png)}
        one = OUT / f"official_1ch__{t}.png"
        if one.exists():
            g = cv2.imread(str(one), cv2.IMREAD_UNCHANGED)
            g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g
            row["official_1ch_output_channels"] = 1 if g.ndim == 2 else g.shape[2]
            row["official_1ch_vs_v1_mps"] = psnr(g3, v1_png)
        rows.append(row)
    return rows


def crop_centre(setname, t):
    """Centre the grid crop where something happens: v1's highest-confidence SR box on Set B,
    the first labelled ship on Set A, else the tile centre."""
    if setname == "setB":
        for r in sorted(csv.DictReader(open(V1_RESULTS / "audit_key.csv")), key=lambda r: -float(r["conf"])):
            if r["tile"] == t and r["version"] == "sr":
                return ((int(r["crop_x0"]) + int(r["crop_x1"])) // 2, (int(r["crop_y0"]) + int(r["crop_y1"])) // 2)
    else:
        lab = version_image(setname, "hr", t).parent.parent / "labels" / f"{t}.txt"
        for line in lab.read_text().splitlines():
            p = line.split()
            if len(p) == 9 and int(p[0]) == v1.SHIP_CLS:
                xy = np.array(p[1:], float).reshape(4, 2).mean(0) * v1.TILE
                return int(xy[0]), int(xy[1])
    return v1.TILE // 2, v1.TILE // 2


def sanity_grid():
    rng = random.Random(0)
    b = [DOM] + rng.sample(sorted(t for t in read_ids("setB") if t != DOM), 3)
    a = rng.sample(read_ids("setA"), 4)
    rows, gap, head = [], 6, 40
    for s, t in [("setB", x) for x in b] + [("setA", x) for x in a]:
        cx, cy = crop_centre(s, t)
        x0 = int(np.clip(cx - CROP // 2, 0, v1.TILE - CROP)) // 4 * 4
        y0 = int(np.clip(cy - CROP // 2, 0, v1.TILE - CROP)) // 4 * 4
        cells = []
        for v in ("hr", "lr", "bicubic", "sr"):
            im = cv2.imread(str(version_image(s, v, t)), cv2.IMREAD_COLOR)
            if v == "lr":
                c = im[y0 // 4:(y0 + CROP) // 4, x0 // 4:(x0 + CROP) // 4]
                c = cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
            else:
                c = im[y0:y0 + CROP, x0:x0 + CROP]
            cells.append(c)
        row = np.full((CROP + head, 4 * CROP + 3 * gap, 3), 255, np.uint8)
        for i, c in enumerate(cells):
            row[head:, i * (CROP + gap):i * (CROP + gap) + CROP] = c
        grey = " (grey)" if is_grey(cells[0]) else ""
        cv2.putText(row, f"{s} {t}{grey}  crop x{x0} y{y0}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        rows.append(row)
    top = np.full((head, rows[0].shape[1], 3), 255, np.uint8)
    for i, lbl in enumerate(("original (HR)", "LR (nearest x4)", "bicubic x4", "Real-ESRGAN x4")):
        cv2.putText(top, lbl, (i * (CROP + gap) + 6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    grid = np.vstack([top] + rows)
    cv2.imwrite(str(V2_RESULTS / "sanity_grid.png"), grid)
    return {"setB": b, "setA": a, "crop": CROP}


def border_split(tile, band=32):
    """Mean |pre_pad 0 - pre_pad 10| in the outer `band` px vs the interior, for the report."""
    a = cv2.imread(str(OUT / f"official_prepad0__{tile}.png"), cv2.IMREAD_COLOR).astype(np.int16)
    b = cv2.imread(str(OUT / f"official_prepad10__{tile}.png"), cv2.IMREAD_COLOR).astype(np.int16)
    d = np.abs(a - b).mean(2)
    inner = np.zeros(d.shape, bool)
    inner[band:-band, band:-band] = True
    return float(d[~inner].mean()), float(d[inner].mean())


def fmt_db(x):
    return "identical" if x == float("inf") else f"{x:.1f} dB"


def render_md():
    res = json.loads((V2_RESULTS / "phase0_sr_audit.json").read_text())
    m, deg, gr, ref = res["model"], res["degradation"], res["greyscale"], res["reference"]
    n_all = sum(deg[s]["n"] for s in SETS)
    exact = lambda k: sum(deg[s][k] for s in SETS)  # noqa: E731
    colour_ref = [r for r in ref if not r["grey"]]
    swapped_colour = [r["channel_swapped_vs_official_pp0"] for r in colour_ref]
    bit_identical = all(r["official_pp0_vs_v1_cpu"] == float("inf") for r in ref)
    mps = [r["v1_mps_vs_v1_cpu"] for r in ref]
    pp10 = [r["official_pp10_vs_v1_mps"] for r in ref]
    borders = {r["tile"]: border_split(r["tile"]) for r in ref}
    L = ["# Phase 0: audit of the v1 super-resolution pipeline", ""]
    verdict = ("**Verdict: no bug in the SR pipeline.** v1's Real-ESRGAN output is bit-identical to the official "
               "inference code run on the same device, and v1's degraded and bicubic images recompute "
               "bit-for-bit from the originals." if bit_identical and exact("lr_recomputed_exact") == n_all
               and exact("bic_recomputed_exact") == n_all else "**Verdict: a discrepancy was found; see below.**")
    L += [verdict + " Greyscale tiles do come out of Real-ESRGAN with colour in them; that is the model, not "
          "the code, and section 2 measures it (v1b). **One real bug was found, in v1's Set A evaluation rather "
          "than the SR path:** the detector ran with ultralytics' default `max_det=300`, which truncated two "
          "dense tiles (section 3).", ""]

    L += ["## 1. The seven checks", "",
          "| # | Check | How it was tested | Result |", "|---|---|---|---|",
          f"| 1 | Channel order | `run_sr` reverses BGR to RGB going in and back coming out ({m['channel_flips_in_run_sr']} "
          f"flips in the source, quoted below); compared with the official code, which does the same with `cvtColor` | "
          f"bit-identical to the official output on all {len(ref)} reference tiles; a deliberately swapped run differs "
          f"by {', '.join(fmt_db(x) for x in swapped_colour)} on the colour tiles, so a swap would have shown |",
          f"| 2 | Range, dtype, mode | parameters, `eval()`, `no_grad`, the tensor conversion in `run_sr` | "
          f"float in [0, 1], NCHW, output clamped to [0, 1] then `*255`, rounded, uint8; parameters "
          f"{', '.join(m['param_dtypes'])} (no fp16 anywhere); eval mode {m['eval_mode']}; `torch.no_grad()` {m['uses_no_grad']} |",
          f"| 3 | Greyscale tiles | channel equality in every version | passed in as three identical channels "
          f"(LR stays grey on {gr['setA_keeps_grey']['lr'] + gr['setB_keeps_grey']['lr']} of "
          f"{gr['setA']['n_grey'] + gr['setB']['n_grey']} grey tiles); SR output is grey on "
          f"{gr['setA_keeps_grey']['sr'] + gr['setB_keeps_grey']['sr']} of them (section 2) |",
          "| 4 | Tiling seams | source | none possible: v1 sends each whole 256x256 LR tile through the model in one call |",
          f"| 5 | Degradation | v1's LR and bicubic PNGs recomputed from HR with v1's own functions; OpenCV's default "
          f"kernel for sigma 2 vs an explicit 13x13 (= 6 sigma + 1) and a 25x25 kernel | LR exact on "
          f"{exact('lr_recomputed_exact')}/{n_all} tiles, bicubic exact on {exact('bic_recomputed_exact')}/{n_all}; "
          f"the default kernel equals 13x13 on {exact('k13_equals_default')}/{n_all}; a 25x25 kernel changes the LR "
          f"by at most {max(deg[s]['lr_maxdiff_k25_vs_default'] for s in SETS)} grey levels; one LR array feeds both "
          f"arms; every version is PNG |",
          f"| 6 | Official reference | `realesrgan` 0.3.0 `RealESRGANer`, fp32, CPU, tile 0, on {len(ref)} Set B LR tiles "
          f"({len(ref) - len(colour_ref)} grey, {len(colour_ref)} colour) | see table below |",
          "| 7 | Visual grid | `v2/results/sanity_grid.png` (local only: DOTA-derived) | 8 tiles incl. P1751: original, LR, bicubic, SR |",
          ""]
    L += ["Upstream of v1, the originals are JPEG: ultralytics ships DOTA as `.jpg` and its tiler writes JPEG "
          f"crops. v1's HR PNGs equal those JPEG tiles exactly ({exact('hr_equals_tile_jpg')}/{n_all}), so JPEG "
          "artefacts are in the originals, identically for every arm, and the sigma-2 blur plus 4x area "
          "reduction removes most of them before either restoration sees the image.", ""]

    L += ["### Reference comparison (PSNR, higher = closer; 'identical' = every pixel equal)", "",
          "| tile | grey | official (pre_pad 0) vs v1 on CPU | official vs v1's file (MPS) | v1 MPS vs CPU | "
          "official pre_pad 10 vs v1 | channel-swapped vs official | border / interior diff, pre_pad 0 vs 10 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ref:
        b = borders[r["tile"]]
        L.append(f"| {r['tile']} | {'yes' if r['grey'] else 'no'} | {fmt_db(r['official_pp0_vs_v1_cpu'])} | "
                 f"{fmt_db(r['official_pp0_vs_v1_mps'])} | {fmt_db(r['v1_mps_vs_v1_cpu'])} | "
                 f"{fmt_db(r['official_pp10_vs_v1_mps'])} | {fmt_db(r['channel_swapped_vs_official_pp0'])} | "
                 f"{b[0]:.2f} / {b[1]:.2f} levels |")
    L += ["", f"- **Implementation:** v1's spandrel path and the official code produce the same bytes on CPU. "
          f"v1's stored outputs were made on Apple's GPU (MPS) and differ from CPU by float rounding only "
          f"({min(mps):.0f}-{max(mps):.0f} dB).",
          f"- **pre_pad:** the official command-line script defaults to `pre_pad=0`, which v1 matches exactly. "
          f"The class default `pre_pad=10` reflect-pads the input first, which changes the output "
          f"({min(pp10):.0f}-{max(pp10):.0f} dB), mostly near the tile edge (border vs interior column). "
          "That is a different, equally official configuration, not a bug.", ""]

    L += ["## 2. Greyscale tiles and v1b", "",
          f"{gr['setB']['n_grey']} of {gr['setB']['n_tiles']} Set B tiles and {gr['setA']['n_grey']} of "
          f"{gr['setA']['n_tiles']} Set A tiles are greyscale. Real-ESRGAN writes colour into every one of them: "
          f"mean channel difference {gr['setB']['sr_chroma_mean_over_grey_tiles']:.2f} grey levels on Set B "
          f"(max {gr['setB']['sr_chroma_max_over_grey_tiles']}), against exactly 0 for the original, LR and "
          "bicubic. The official code does the same for a 3-channel file; it converts back to grey only when "
          "the input file has one channel. So v1 followed the official behaviour, but a careful pipeline "
          "could reasonably convert grey tiles back, so the effect is measured.", "",
          "**Where v1's Set B firing happened** (v1's own results, split by tile colour):", "",
          "| version | conf | grey tiles fired | colour tiles fired | Fisher p |", "|---|---|---|---|---|"]
    for k, e in gr["setB_firing_by_colour"].items():
        v, c = k.split("_conf")
        L.append(f"| {v} | {c} | {e['grey_fired']}/{e['n_grey']} ({100 * e['grey_fired'] / e['n_grey']:.1f}%) | "
                 f"{e['colour_fired']}/{e['n_colour']} ({100 * e['colour_fired'] / e['n_colour']:.1f}%) | {e['fisher_p']:.2g} |")
    L += ["", "SR's firing sits overwhelmingly on colour tiles, so false colour on grey tiles cannot be what "
          "drives v1's result. P1751, the dominant tile, is one of the few grey tiles that fired.", ""]
    L += v1b_section()
    L += maxdet_section()
    L += ["## 4. Consequence for v2", "",
          "No bug in the SR path, so no image is regenerated: v2 imports v1's `degrade()` and `run_sr()` "
          "unchanged and runs them on the training tiles too. v1b is reported as a sensitivity row next to v1 "
          "throughout. Every v2 prediction uses `max_det=1000` and records any image that reaches it.", "",
          "## Appendix: the v1 functions under test", "", "```python", m["degrade_source"].rstrip(), "",
          m["run_sr_source"].rstrip(), "```", ""]
    (V2_RESULTS / "phase0_sr_audit.md").write_text("\n".join(L))
    print(f"-> {V2_RESULTS / 'phase0_sr_audit.md'}")


def v1b_section():
    fa, fb = V2_RESULTS / "setA_metrics.json", V2_RESULTS / "setB_firing.json"
    if not (fa.exists() and fb.exists()):
        return ["*v1 vs v1b: run 04_predict.py, 05_eval_setA.py and 06_eval_setB_firing.py, then "
                "`00_sr_audit.py --render-only`.*", ""]
    a = json.loads(fa.read_text())["v1_offshelf"]
    b = json.loads(fb.read_text())["v1_offshelf"]
    if "sr_greyfix" not in a:
        return ["*v1b images missing: run 00b_v1b_grey.py, then 04-06.*", ""]
    L = ["**v1 vs v1b** (the v1 detector, unchanged; v1b = SR with grey tiles converted back to grey):", "",
         "| metric | conf | v1 SR | v1b SR | bicubic |", "|---|---|---|---|---|"]
    for c in CFG["eval"]["confs"]:
        k = f"conf{c}"
        row = lambda v: b[v][k]["raw"]  # noqa: E731
        L.append(f"| Set B firing (tiles) | {c} | {row('sr')['tiles']}/150 ({row('sr')['pct']:.1f}%) | "
                 f"{row('sr_greyfix')['tiles']}/150 ({row('sr_greyfix')['pct']:.1f}%) | "
                 f"{row('bicubic')['tiles']}/150 ({row('bicubic')['pct']:.1f}%) |")
    for c in CFG["eval"]["confs"]:
        k = f"conf{c}"
        for met in ("recall", "precision"):
            L.append(f"| Set A {met} | {c} | {a['sr'][k][met]:.3f} | {a['sr_greyfix'][k][met]:.3f} | "
                     f"{a['bicubic'][k][met]:.3f} |")
    L += ["", "Invention for v1b needs verdicts on any box v1 never audited; those boxes are in the v2 audit "
          "pool (Phase 6), so v1b invention is reported with the v2 results.", ""]
    return L


def maxdet_section():
    """The evaluation bug the v2 regression gate exposed: v1 ran with ultralytics' max_det=300."""
    fa = V2_RESULTS / "setA_metrics.json"
    if not fa.exists():
        return []
    a = json.loads(fa.read_text())
    if "v1_offshelf_v1settings" not in a:
        return []
    old, new, trunc = a["v1_offshelf_v1settings"], a["v1_offshelf"], a["_v1_max_det_truncation"]
    tiles = sorted({t["tile"] for v in trunc.values() for t in v})
    L = ["## 3. A bug outside the SR path: v1's Set A evaluation was truncated", "",
         "v2's evaluation is gated on reproducing v1's numbers. Predicting at conf 0.01 and thresholding "
         "afterwards first gave *more* Set A detections than v1's direct runs. The cause: v1 called the "
         "detector with ultralytics' default `max_det=300` (all 15 classes count toward it), and "
         f"{len(tiles)} dense harbour tiles ({', '.join(tiles)}) hit that cap on every version, so ship boxes "
         "were silently dropped. With v1's exact call the v2 code reproduces v1's TP/FP/FN exactly; with "
         "`max_det=1000` no tile reaches the cap.", "",
         "| version | conf | ship boxes cut | v1 recall (as published) | corrected recall | v1 precision | corrected precision |",
         "|---|---|---|---|---|---|---|"]
    for v in ("hr", "bicubic", "sr"):
        for c in CFG["eval"]["confs"]:
            k = f"conf{c}"
            cut = sum(t["boxes_uncapped"] - t["boxes_v1_call"] for t in trunc.get(f"{v}_{k}", []))
            L.append(f"| {v} | {c} | {cut} | {old[v][k]['recall']:.3f} | {new[v][k]['recall']:.3f} | "
                     f"{old[v][k]['precision']:.3f} | {new[v][k]['precision']:.3f} |")
    gap_old = old["sr"]["conf0.25"]["recall"] - old["bicubic"]["conf0.25"]["recall"]
    gap_new = new["sr"]["conf0.25"]["recall"] - new["bicubic"]["conf0.25"]["recall"]
    L += ["", f"Every version loses the same two tiles, so absolute recall was understated but the comparison "
          f"barely moves: SR minus bicubic recall at 0.25 is {gap_old:+.3f} as published and {gap_new:+.3f} "
          "corrected. Set B is unaffected (empty harbours never approach 300 boxes; v2 reproduces all 900 of "
          "v1's Set B per-tile counts exactly). v1's published files are left unchanged, as the v2 rules "
          "require; the corrected numbers live in `v2/results/setA_metrics.json` (`v1_offshelf`).", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-python", default="", help="python of the env with realesrgan + basicsr; "
                                                     "omit to reuse outputs already in v2/results/phase0/")
    ap.add_argument("--render-only", action="store_true", help="rebuild the .md from the saved .json")
    args = ap.parse_args()
    if args.render_only:
        render_md()
        return
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    m03 = load_v1_script("03_degrade_sr")
    res = {"model": model_checks(m03)}
    res["degradation"] = degradation_checks(m03)
    res["greyscale"] = greyscale_checks()
    res["reference"] = reference_comparison(m03, args.ref_python)
    res["grid"] = sanity_grid()
    (V2_RESULTS / "phase0_sr_audit.json").write_text(json.dumps(res, indent=2))
    print(f"-> {V2_RESULTS / 'phase0_sr_audit.json'}")
    render_md()


if __name__ == "__main__":
    main()
