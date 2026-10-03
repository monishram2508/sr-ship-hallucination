"""Set A precision/recall/F1 for the ship class at a FIXED confidence, identical for all versions.

Why this exists: ultralytics reports P and R at the confidence maximising the smoothed MEAN F1
across all 15 classes, and that confidence differs per version (HR 0.627, bicubic 0.338, SR 0.565).
Comparing recall between versions at different thresholds is not a comparison. This re-measures
all three at the same bar, by matching detections to ground-truth ship polygons with rotated-box
IoU (ultralytics' own batch_probiou), greedy, highest-confidence first.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, DETECTOR, IMGSZ, SHIP_CLS, TILE, device  # noqa: E402

VERSIONS = ("hr", "bicubic", "sr")
CONFS = (0.25, 0.5)
IOU_T = 0.5


def gt_ship_xywhr(tile):
    """Ground-truth ship polygons -> (N,5) xywhr in pixels."""
    from ultralytics.utils.ops import xyxyxyxy2xywhr
    p = DATA / "versions" / "hr" / "setA" / "labels" / f"{tile}.txt"
    polys = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) != 9 or int(parts[0]) != SHIP_CLS:
            continue
        polys.append([float(x) for x in parts[1:]])
    if not polys:
        return np.zeros((0, 5), np.float32)
    arr = np.asarray(polys, np.float32).reshape(-1, 4, 2) * TILE
    return np.asarray(xyxyxyxy2xywhr(arr), np.float32).reshape(-1, 5)


def evaluate(model, dev, version, conf):
    imgs = sorted((DATA / "versions" / version / "setA" / "images").glob("*.png"))
    TP = FP = FN = 0
    for i in range(0, len(imgs), 8):
        group = imgs[i:i + 8]
        for p, r in zip(group, model.predict([str(x) for x in group], imgsz=IMGSZ, conf=conf,
                                             device=dev, verbose=False)):
            gt = gt_ship_xywhr(p.stem)
            if r.obb is None or not len(r.obb):
                det = np.zeros((0, 5), np.float32)
                dc = np.zeros((0,), np.float32)
            else:
                keep = (r.obb.cls == SHIP_CLS).cpu().numpy()
                det = r.obb.xywhr.cpu().numpy()[keep]
                dc = r.obb.conf.cpu().numpy()[keep]
            order = np.argsort(-dc)
            det = det[order]
            if len(gt) == 0:
                FP += len(det)
                continue
            if len(det) == 0:
                FN += len(gt)
                continue
            iou = batch_iou(gt, det)                      # (n_gt, n_det)
            used = np.zeros(len(gt), bool)
            for j in range(len(det)):
                col = iou[:, j].copy()
                col[used] = -1
                k = int(col.argmax())
                if col[k] >= IOU_T:
                    used[k] = True
                    TP += 1
                else:
                    FP += 1
            FN += int((~used).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"conf": conf, "TP": TP, "FP": FP, "FN": FN,
            "precision": prec, "recall": rec, "f1": f1}


def batch_iou(gt, det):
    from ultralytics.utils.metrics import batch_probiou
    return np.asarray(batch_probiou(torch.from_numpy(gt), torch.from_numpy(det)))


def main():
    from ultralytics import YOLO
    dev, model = device(), YOLO(DETECTOR)
    out = {"iou_threshold": IOU_T, "note": "greedy highest-confidence-first matching, rotated-box "
                                           "IoU via ultralytics batch_probiou; ship class only",
           "results": {}}
    print(f"{'version':12s} {'conf':>5s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for c in CONFS:
        for v in VERSIONS:
            r = evaluate(model, dev, v, c)
            out["results"][f"{v}_conf{c}"] = r
            print(f"{v:12s} {c:5.2f} {r['TP']:6d} {r['FP']:6d} {r['FN']:6d} "
                  f"{r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f}")
    (RESULTS / "setA_fixed_threshold.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RESULTS/'setA_fixed_threshold.json'}")


if __name__ == "__main__":
    main()
