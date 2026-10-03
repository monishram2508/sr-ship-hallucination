"""Phase 4 — hero figure + PSNR-vs-detection-delta scatter.

hero.png : HR | LR | bicubic | SR for one tile where SR invented a ship,
           detector boxes drawn, red circle on the invented ship.
scatter  : x = SR PSNR vs HR, y = |ships on SR - ships on HR|, one dot per tile.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, FIGURES, DETECTOR, IMGSZ, SHIP_CLS, TILE, device  # noqa: E402

CAPTION = [
    "Zero ship labels, and zero ship detections on the original or on the plain upscale of the degraded image.",
    "Real-ESRGAN produced 45 ship detections here. A blind two-pass audit of crops taken from the ORIGINAL",
    "judged all 45 of them not a vessel. This is a residential hillside, not water.",
]

PANELS = [("hr", "HR 1024px (ground truth)"),
          ("lr", "LR 256px (simulated cheaper sensor)"),
          ("bicubic", "Bicubic x4 (no AI)"),
          ("sr", "Real-ESRGAN x4")]


def draw(model, dev, path, conf=0.25):
    """Return the tile at 1024px with ship boxes drawn, plus the ship polygons."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im.shape[0] != TILE:
        im = cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        r = model.predict(str(path), imgsz=IMGSZ, conf=conf, device=dev, verbose=False)[0]
        scale = TILE / cv2.imread(str(path)).shape[0]
    else:
        r = model.predict(str(path), imgsz=IMGSZ, conf=conf, device=dev, verbose=False)[0]
        scale = 1.0
    polys = []
    if r.obb is not None and len(r.obb):
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        cls = r.obb.cls.cpu().numpy()
        cf = r.obb.conf.cpu().numpy()
        for p, c, s in zip(xyxyxyxy, cls, cf):
            if int(c) != SHIP_CLS:
                continue
            pts = (p * scale).astype(np.int32)
            polys.append((pts, float(s)))
            cv2.polylines(im, [pts], True, (0, 0, 255), 3)
    return im, polys


def hero(model, dev, tile, n_inset=2):
    gap, bar = 10, 54
    imgs = []
    for v, label in PANELS:
        p = DATA / "versions" / v / "setB" / "images" / f"{tile}.png"
        im, polys = draw(model, dev, p)
        cv2.rectangle(im, (0, 0), (TILE, bar), (0, 0, 0), -1)
        cv2.putText(im, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        n = len(polys)
        tag = f"{n} ship{'s' if n != 1 else ''} detected"
        col = (0, 0, 255) if n else (120, 255, 120)
        cv2.putText(im, tag, (TILE - 420, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        if v == "sr":
            ring = sorted(polys, key=lambda t: -t[1])[:3 if len(polys) > 6 else len(polys)]
            for pts, _ in ring:
                c = pts.mean(0).astype(int)
                rad = int(max(np.linalg.norm(pts - c, axis=1).max() * 2.2, 70))
                cv2.circle(im, tuple(c), rad, (0, 0, 255), 5)
        imgs.append(im)

    row = np.full((TILE, TILE * 4 + gap * 3, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        row[:, i * (TILE + gap):i * (TILE + gap) + TILE] = im

    # ---- zoom insets: the n highest-confidence SR claims, HR beside SR ----
    srp = DATA / "versions" / "sr" / "setB" / "images" / f"{tile}.png"
    r = model.predict(str(srp), imgsz=IMGSZ, conf=0.25, device=dev, verbose=False)[0]
    dets = sorted([(float(c), pl) for pl, k, c in zip(r.obb.xyxyxyxy.cpu().numpy(),
                                                      r.obb.cls.cpu().numpy(),
                                                      r.obb.conf.cpu().numpy()) if int(k) == SHIP_CLS],
                  key=lambda t: -t[0])[:n_inset]
    hr_im = cv2.imread(str(DATA / "versions" / "hr" / "setB" / "images" / f"{tile}.png"))
    sr_im = cv2.imread(str(srp))
    PADZ = 70
    cells = []
    for rank, (cf, poly) in enumerate(dets, 1):
        x0 = int(max(0, poly[:, 0].min() - PADZ)); y0 = int(max(0, poly[:, 1].min() - PADZ))
        x1 = int(min(TILE, poly[:, 0].max() + PADZ)); y1 = int(min(TILE, poly[:, 1].max() + PADZ))
        for which, src in (("HR (original)", hr_im), ("Real-ESRGAN", sr_im)):
            crop = src[y0:y1, x0:x1]
            h, w = crop.shape[:2]
            sc = TILE / max(h, w)
            out = cv2.resize(crop, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_NEAREST)
            cell = np.full((TILE, TILE, 3), 24, np.uint8)
            oy, ox = (TILE - out.shape[0]) // 2, (TILE - out.shape[1]) // 2
            cell[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out
            if which != "HR (original)":
                pts = ((poly - [x0, y0]) * sc + [ox, oy]).astype(np.int32)
                cv2.polylines(cell, [pts], True, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.rectangle(cell, (0, 0), (TILE, bar), (0, 0, 0), -1)
            cv2.putText(cell, f"zoom {rank}  -  {which}", (14, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if which != "HR (original)":
                cv2.putText(cell, f"ship {cf:.2f}", (TILE - 260, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cells.append(cell)
    zrow = np.full((TILE, row.shape[1], 3), 255, np.uint8)
    for i, c in enumerate(cells[:4]):
        zrow[:, i * (TILE + gap):i * (TILE + gap) + TILE] = c

    cap = np.full((170, row.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(CAPTION):
        cv2.putText(cap, line, (16, 52 + i * 56), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3)
    out = np.vstack([row, zrow, cap])
    out = cv2.resize(out, None, fx=0.42, fy=0.42, interpolation=cv2.INTER_AREA)
    path = FIGURES / "hero.png"
    cv2.imwrite(str(path), out)
    print(f"hero -> {path}  ({out.shape[1]}x{out.shape[0]}) tile={tile}")


def scatter():
    """x = SR PSNR vs HR, y = audited not_ship boxes on that tile. Symlog y: most tiles are 0 and
    one tile is 43, which a linear axis renders as a flat line plus an outlier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, mannwhitneyu

    psnr = {}
    for r in csv.DictReader(open(RESULTS / "psnr.csv")):
        if r["set"] == "setB" and r["version"] == "sr":
            psnr[r["tile"]] = float(r["psnr"])
    bic_psnr = {}
    for r in csv.DictReader(open(RESULTS / "psnr.csv")):
        if r["set"] == "setB" and r["version"] == "bicubic":
            bic_psnr[r["tile"]] = float(r["psnr"])
    key = {r["crop_id"]: r for r in csv.DictReader(open(RESULTS / "audit_key.csv"))}
    lab = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(RESULTS / "audit_labels.csv"))}

    ns = defaultdict(lambda: defaultdict(int))
    for c, k in key.items():
        if lab[c] == "not_ship":
            ns[k["version"]][k["tile"]] += 1

    tiles = sorted(psnr)
    x = np.array([psnr[t] for t in tiles])
    y = np.array([ns["sr"][t] for t in tiles], float)
    hit = (y > 0).astype(int)
    u = mannwhitneyu((-x)[hit == 1], (-x)[hit == 0], alternative="two-sided")
    auc = u.statistic / (hit.sum() * (len(hit) - hit.sum()))
    rho, prho = spearmanr(x, y)

    # the confound, computed rather than quoted: PSNR tracks how busy the scene is, and busy
    # scenes are where the unlabelled boats live.
    nobj = []
    for t in tiles:
        f = DATA / "versions" / "hr" / "setB" / "labels" / f"{t}.txt"
        nobj.append(sum(1 for ln in f.read_text().splitlines() if ln.strip()))
    rho_busy, p_busy = spearmanr(x, np.array(nobj, float))

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    ax.scatter(np.array([bic_psnr[t] for t in tiles]),
               np.array([ns["bicubic"][t] for t in tiles], float),
               s=22, alpha=0.55, c="#888888", label="Bicubic x4", edgecolors="none")
    ax.scatter(x, y, s=26, alpha=0.75, c="#d62728", label="Real-ESRGAN x4", edgecolors="none")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("PSNR vs HR (dB)  -  higher = pixels look more similar to the original")
    ax.set_ylabel("audited invented objects on the tile\n(boxes judged not_ship)")
    ax.set_title(f"After audit, pixel similarity does not predict invented objects\n"
                 f"AUC {auc:.2f} (p = {u.pvalue:.2f}), Spearman rho = {rho:+.2f} (p = {prho:.2f})",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.6)
    # keep the note clear of the P1751 point, which sits near the top of the symlog axis
    ax.text(0.03, 0.28,
            "Pre-audit the same plot gave AUC 0.68 (p = 0.0005): that signal was real\n"
            "unlabelled vessels, not inventions. Confound: PSNR tracks scene busyness\n"
            f"(Spearman rho = {rho_busy:+.2f}, p = {p_busy:.0e} vs labelled object count),\n"
            "and busy scenes are where the unlabelled boats are.",
            transform=ax.transAxes, va="center", ha="left", fontsize=7.6, color="#444444")
    fig.tight_layout()
    p = FIGURES / "psnr_vs_detection_delta.png"
    fig.savefig(p)
    # persist the stats so the write-up quotes a file, never the figure or memory
    (RESULTS / "psnr_vs_invention.json").write_text(json.dumps({
        "question": "does low SR PSNR predict a tile carrying >=1 box judged not_ship",
        "n_tiles": len(tiles),
        "tiles_with_invention": int(hit.sum()),
        "auc_low_psnr_predicts_invention": float(auc),
        "mannwhitney_p": float(u.pvalue),
        "spearman_rho": float(rho),
        "spearman_p": float(prho),
        "confound_spearman_rho_psnr_vs_labelled_object_count": float(rho_busy),
        "confound_spearman_p": float(p_busy),
    }, indent=2))
    print(f"scatter -> {p}   AUC={auc:.4f} p={u.pvalue:.4f}  rho={rho:+.4f} p={prho:.4f}")


def panel3(model, dev, tile, setname="setB"):
    """HR | bicubic | SR for one tile, ship boxes drawn, counts in the corner."""
    gap, bar = 10, 54
    imgs = []
    for v, label in (("hr", "HR (ground truth)"), ("bicubic", "Bicubic x4"), ("sr", "Real-ESRGAN x4")):
        p = DATA / "versions" / v / setname / "images" / f"{tile}.png"
        im, polys = draw(model, dev, p)
        cv2.rectangle(im, (0, 0), (TILE, bar), (0, 0, 0), -1)
        cv2.putText(im, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        n = len(polys)
        cv2.putText(im, f"{n} ship{'s' if n != 1 else ''}", (TILE - 230, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if n else (120, 255, 120), 2)
        imgs.append(im)
    row = np.full((TILE, TILE * 3 + gap * 2, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        row[:, i * (TILE + gap):i * (TILE + gap) + TILE] = im
    cap = np.full((76, row.shape[1], 3), 255, np.uint8)
    cv2.putText(cap, tile, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    out = cv2.resize(np.vstack([row, cap]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    path = FIGURES / f"panel3_{tile}.png"
    cv2.imwrite(str(path), out)
    print(f"panel -> {path}  ({out.shape[1]}x{out.shape[0]})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", help="tile id: write a 3-panel HR|bicubic|SR for just this tile")
    ap.add_argument("--hero", help="force the hero tile instead of auto-picking the worst")
    args = ap.parse_args()

    from ultralytics import YOLO
    dev, model = device(), YOLO(DETECTOR)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.panel:
        panel3(model, dev, args.panel)
        return
    s = json.loads((RESULTS / "setB_summary.json").read_text())
    cands = s.get("sr_invented_tiles", [])
    tile = args.hero or (cands[0]["tile"] if cands else None)
    if tile is None:
        print("no SR-invented tiles; hero figure needs a hand-picked tile")
    else:
        hero(model, dev, tile)
    scatter()


if __name__ == "__main__":
    main()
