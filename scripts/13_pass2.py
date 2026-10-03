"""Build the pass-2 adjudication set: the crops labelled `unsure` in pass 1, re-rendered with
much more context (128 px pad instead of 48) so a second look has something to go on.

Still blind: crop_id only, and the presentation order is reshuffled with a NEW seed so pass-1
ordering carries over no information. crop_ids are unchanged, which is what lets the two passes
merge. Pass-1 labels live in results/audit_labels_pass1.csv and are never touched.
"""
import csv, random, sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, FIGURES, DETECTOR, IMGSZ, SHIP_CLS, device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location("a8", Path(__file__).parent / "08_audit_boxes.py")
_a8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a8)

PAD2 = 128
BIG = 900
SEED2 = 1
PASS1 = RESULTS / "audit_labels_pass1.csv"
KEY = RESULTS / "audit_key.csv"
OUT_LABELS = RESULTS / "audit_labels_pass2.csv"
CROPS2 = FIGURES / "audit" / "crops_pass2"


def main():
    key = {r["crop_id"]: r for r in csv.DictReader(open(KEY))}
    pass1 = {r["crop_id"]: r["verdict"] for r in csv.DictReader(open(PASS1))}
    targets = sorted(c for c, v in pass1.items() if v == "unsure")
    if not targets:
        sys.exit("no `unsure` crops in pass 1 — nothing to adjudicate")
    print(f"{len(targets)} crops to re-adjudicate")

    if OUT_LABELS.exists():
        done = [r for r in csv.DictReader(open(OUT_LABELS)) if (r.get("verdict") or "").strip()]
        if done:
            sys.exit(f"REFUSING: {OUT_LABELS} already holds {len(done)} verdicts.")

    need = {}
    for c in targets:
        k = key[c]
        need.setdefault((k["version"], k["tile"]), {})[int(k["box_idx"])] = c

    from ultralytics import YOLO
    dev, model = device(), YOLO(DETECTOR)
    CROPS2.mkdir(parents=True, exist_ok=True)
    for old in CROPS2.glob("*.png"):
        old.unlink()

    _a8.PAD = PAD2          # same renderer, wider context
    made = 0
    for (version, tile), boxes in tqdm(sorted(need.items()), desc="pass-2 crops"):
        p = DATA / "versions" / version / "setB" / "images" / f"{tile}.png"
        r = model.predict(str(p), imgsz=IMGSZ, conf=0.25, device=dev, verbose=False)[0]
        dets = sorted([(float(cf), pl) for pl, k, cf in zip(r.obb.xyxyxyxy.cpu().numpy(),
                                                            r.obb.cls.cpu().numpy(),
                                                            r.obb.conf.cpu().numpy())
                       if int(k) == SHIP_CLS], key=lambda t: -t[0])
        hr = cv2.imread(str(DATA / "versions" / "hr" / "setB" / "images" / f"{tile}.png"),
                        cv2.IMREAD_COLOR)
        for bi, cid in boxes.items():
            cf, poly = dets[bi]
            assert abs(cf - float(key[cid]["conf"])) < 1e-3, \
                f"{cid}: confidence drifted ({cf} vs {key[cid]['conf']}) — box identity is not stable"
            cv2.imwrite(str(CROPS2 / f"{cid}.png"),
                        _a8.render(hr, {"poly": poly}, BIG, 2))
            made += 1

    order = list(targets)
    random.Random(SEED2).shuffle(order)
    with open(OUT_LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "verdict"])
        for c in order:
            w.writerow([c, ""])

    print(f"\n{made} crops at pad {PAD2} px -> {CROPS2}")
    print(f"order reshuffled with seed {SEED2}; blank verdicts -> {OUT_LABELS}")
    print(f"pass-1 labels preserved read-only at {PASS1}")
    print("\nnext: .venv/bin/python scripts/11_label.py --pass2")


if __name__ == "__main__":
    main()
