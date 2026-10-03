"""Build the BLINDED manual audit: crops, contact sheets, and the two split CSVs.

Every ship detection on Set B at conf 0.25 -- SR and bicubic both -- is cropped from the **HR**
tile (pad 48 px, INTER_NEAREST so no interpolation can invent structure) with a thin magenta
outline marking only WHERE the claim is. Judging on HR, not on SR, is what keeps the judgement
uncontaminated by what the GAN drew.

BLINDING: crop order is shuffled with a fixed seed and crop_ids are assigned AFTER the shuffle, so
an id carries no information about version, tile or confidence. The mapping lives in
results/audit_key.csv, which the labeller never opens; the labeller reads and writes only
results/audit_labels.csv (crop_id, verdict). 09_recount.py joins the two.

Bicubic's boxes are audited alongside SR's: removing real objects from one arm of a paired
comparison and not the other would invalidate the comparison.
"""
import argparse, csv, random, sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, FIGURES, DETECTOR, IMGSZ, SHIP_CLS, TILE, device  # noqa: E402

PAD = 48
CELL = 300          # contact-sheet cell
BIG = 900           # per-crop image used by the labeller
CAP = 28
COLS, ROWS = 5, 4
PER_SHEET = COLS * ROWS
CONF = 0.25
SEED = 0
AUDIT = FIGURES / "audit"
CROPS = AUDIT / "crops"
KEY = RESULTS / "audit_key.csv"
LABELS = RESULTS / "audit_labels.csv"


def collect(model, dev, version):
    imgs = sorted((DATA / "versions" / version / "setB" / "images").glob("*.png"))
    found = []
    for i in range(0, len(imgs), 8):
        group = imgs[i:i + 8]
        for p, r in zip(group, model.predict([str(x) for x in group], imgsz=IMGSZ, conf=CONF,
                                             device=dev, verbose=False)):
            if r.obb is None or not len(r.obb):
                continue
            poly = r.obb.xyxyxyxy.cpu().numpy()
            cls = r.obb.cls.cpu().numpy()
            cf = r.obb.conf.cpu().numpy()
            dets = [(float(c), pl) for pl, k, c in zip(poly, cls, cf) if int(k) == SHIP_CLS]
            dets.sort(key=lambda t: -t[0])
            for bi, (c, pl) in enumerate(dets):
                found.append({"version": version, "tile": p.stem, "box_idx": bi,
                              "conf": c, "poly": pl})
    return found


def crop_box(det):
    poly = det["poly"]
    x0 = int(max(0, np.floor(poly[:, 0].min()) - PAD))
    y0 = int(max(0, np.floor(poly[:, 1].min()) - PAD))
    x1 = int(min(TILE, np.ceil(poly[:, 0].max()) + PAD))
    y1 = int(min(TILE, np.ceil(poly[:, 1].max()) + PAD))
    return x0, y0, x1, y1


def render(hr, det, size, thickness):
    x0, y0, x1, y1 = crop_box(det)
    crop = hr[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = size / max(h, w)
    out = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 32, np.uint8)
    oy, ox = (size - out.shape[0]) // 2, (size - out.shape[1]) // 2
    canvas[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out
    pts = ((det["poly"] - [x0, y0]) * s + [ox, oy]).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (255, 0, 255), thickness, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite audit_labels.csv even if it already holds verdicts")
    args = ap.parse_args()

    if LABELS.exists() and not args.force:
        done = [r for r in csv.DictReader(open(LABELS)) if (r.get("verdict") or "").strip()]
        if done:
            sys.exit(f"REFUSING: {LABELS} already holds {len(done)} verdicts. "
                     "Rerunning would destroy them. Pass --force if that is really what you want.")

    from ultralytics import YOLO
    dev, model = device(), YOLO(DETECTOR)
    CROPS.mkdir(parents=True, exist_ok=True)
    for old in AUDIT.glob("sheet_*.png"):
        old.unlink()
    for old in CROPS.glob("*.png"):
        old.unlink()

    dets = []
    for version in ("sr", "bicubic"):
        d = collect(model, dev, version)
        print(f"{version}: {len(d)} ship detections at conf {CONF}")
        dets += d

    random.Random(SEED).shuffle(dets)              # blind: shuffle BEFORE numbering
    for i, d in enumerate(dets, 1):
        d["crop_id"] = f"c{i:03d}"

    cells = {}
    cur_tile, hr = None, None
    for d in tqdm(sorted(dets, key=lambda x: x["tile"]), desc="crops"):
        if d["tile"] != cur_tile:
            cur_tile = d["tile"]
            hr = cv2.imread(str(DATA / "versions" / "hr" / "setB" / "images" / f"{cur_tile}.png"),
                            cv2.IMREAD_COLOR)
        cells[d["crop_id"]] = render(hr, d, CELL, 1)
        cv2.imwrite(str(CROPS / f"{d['crop_id']}.png"), render(hr, d, BIG, 2))

    sheets = [dets[i:i + PER_SHEET] for i in range(0, len(dets), PER_SHEET)]
    for si, batch in enumerate(tqdm(sheets, desc="sheets"), 1):
        W = COLS * CELL + (COLS + 1) * 10
        H = ROWS * (CELL + CAP) + (ROWS + 1) * 10 + 50
        sheet = np.full((H, W, 3), 245, np.uint8)
        cv2.putText(sheet, f"Set B box audit  -  HR crops  -  sheet {si}/{len(sheets)}"
                           f"  -  label each: ship / not_ship / unsure",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2)
        for k, d in enumerate(batch):
            rr, cc = divmod(k, COLS)
            x = 10 + cc * (CELL + 10)
            y = 50 + 10 + rr * (CELL + CAP + 10)
            sheet[y:y + CELL, x:x + CELL] = cells[d["crop_id"]]
            cv2.putText(sheet, d["crop_id"], (x + 4, y + CELL + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(AUDIT / f"sheet_{si:02d}.png"), sheet)

    with open(KEY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "version", "tile", "box_idx", "conf",
                    "crop_x0", "crop_y0", "crop_x1", "crop_y1"])
        for d in sorted(dets, key=lambda x: x["crop_id"]):
            bx = crop_box(d)
            w.writerow([d["crop_id"], d["version"], d["tile"], d["box_idx"],
                        round(d["conf"], 4), *bx])
    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "verdict"])
        for d in sorted(dets, key=lambda x: x["crop_id"]):
            w.writerow([d["crop_id"], ""])

    n_sr = sum(1 for d in dets if d["version"] == "sr")
    print(f"\n{len(dets)} crops ({n_sr} SR, {len(dets)-n_sr} bicubic), {len(sheets)} sheets")
    print(f"  crops   -> {CROPS}")
    print(f"  sheets  -> {AUDIT}")
    print(f"  key     -> {KEY}   (do not open while labelling)")
    print(f"  labels  -> {LABELS}")
    print("next: .venv/bin/python scripts/11_label.py")


if __name__ == "__main__":
    main()
