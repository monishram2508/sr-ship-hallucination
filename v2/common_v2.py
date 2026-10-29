"""Shared helpers for v2.

v1 code is imported read-only: scripts/common.py by plain import, and the numbered v1 scripts
(e.g. 03_degrade_sr.py, whose degrade() and run_sr() define the v1 pipeline) through importlib, so
v2 runs the exact v1 functions instead of a copy that could drift.
"""
import csv, hashlib, importlib.util, sys
from pathlib import Path

import numpy as np
import yaml

V2 = Path(__file__).resolve().parent
ROOT = V2.parent
sys.path.insert(0, str(ROOT / "scripts"))
import common as v1  # noqa: E402  v1 scripts/common.py, never edited

CFG = yaml.safe_load((V2 / "config.yaml").read_text())


def repo_path(rel):
    """Config paths are relative to the repository root."""
    return ROOT / rel


V1_VERSIONS = repo_path(CFG["v1"]["versions"])
V1_SETS = repo_path(CFG["v1"]["sets"])
V1_RESULTS = repo_path(CFG["v1"]["results"])
V2_DATA = repo_path(CFG["v2_data"])
V2_RESULTS = repo_path(CFG["v2_results"])
TILE = CFG["tile"]
VERSIONS = ("hr", "bicubic", "sr")
SETS = ("setA", "setB")

# v2 must degrade exactly as v1 did; fail loudly if the config and v1 ever disagree
assert TILE == v1.TILE and CFG["scale"] == v1.SCALE and CFG["blur_sigma"] == v1.BLUR_SIGMA, \
    "v2/config.yaml disagrees with v1 scripts/common.py"

# one fixed colour per condition for every v2 figure (validated categorical slots 1-3, light surface)
CONDITION_COLOURS = {"M_HR": "#2a78d6", "M_SR": "#eb6834", "M_BIC": "#1baf7a", "v1": "#6b6a64"}

PRED_FIELDS = ["model", "condition", "seed", "set", "version", "tile", "conf",
               "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "cx", "cy", "w", "h", "r"]


def load_v1_script(name):
    """Import a numbered v1 script (e.g. '03_degrade_sr') as a module without touching it."""
    p = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"v1_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_ids(setname):
    return [l.strip() for l in (V1_SETS / f"{setname}.txt").read_text().splitlines() if l.strip()]


def version_image(setname, version, tile):
    return V1_VERSIONS / version / setname / "images" / f"{tile}.png"


def is_grey(img):
    """True when all three channels are identical (a greyscale tile stored as BGR)."""
    return bool((img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all())


def array_md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---- rotated boxes --------------------------------------------------------------------------

def poly_to_xywhr(polys):
    """(N,4,2) or (N,8) pixel corners -> (N,5) xywhr."""
    from ultralytics.utils.ops import xyxyxyxy2xywhr
    a = np.asarray(polys, np.float32).reshape(-1, 4, 2)
    if not len(a):
        return np.zeros((0, 5), np.float32)
    return np.asarray(xyxyxyxy2xywhr(a), np.float32).reshape(-1, 5)


def probiou(a, b):
    """(n,5) x (m,5) xywhr -> (n,m) probabilistic IoU (ultralytics' batch_probiou, as in v1)."""
    import torch
    from ultralytics.utils.metrics import batch_probiou
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    return np.asarray(batch_probiou(torch.from_numpy(np.asarray(a, np.float32)),
                                    torch.from_numpy(np.asarray(b, np.float32))))


def gt_xywhr(label_file, cls):
    """Ground-truth polygons of one class from a normalised YOLO-OBB label file -> (N,5) pixels."""
    polys = []
    for line in Path(label_file).read_text().splitlines():
        parts = line.split()
        if len(parts) == 9 and int(parts[0]) == cls:
            polys.append([float(x) for x in parts[1:]])
    if not polys:
        return np.zeros((0, 5), np.float32)
    return poly_to_xywhr(np.asarray(polys, np.float32).reshape(-1, 4, 2) * TILE)


def greedy_tp(gt, det, iou_t):
    """v1's matching (10_fixed_threshold.py): detections already sorted by confidence, highest
    first; each takes the best still-unused ground truth if IoU >= iou_t. Greedy in order, so the
    result for any confidence cut-off is a prefix of this one pass."""
    tp = np.zeros(len(det), bool)
    if not len(gt) or not len(det):
        return tp
    iou = probiou(gt, det)
    used = np.zeros(len(gt), bool)
    for j in range(len(det)):
        col = iou[:, j].copy()
        col[used] = -1
        k = int(col.argmax())
        if col[k] >= iou_t:
            used[k] = True
            tp[j] = True
    return tp


def ap50(tp_flags, confs, n_gt):
    """AP from pooled detections (101-point interpolation, ultralytics' compute_ap)."""
    from ultralytics.utils.metrics import compute_ap
    if n_gt == 0 or not len(confs):
        return 0.0
    order = np.argsort(-np.asarray(confs), kind="stable")
    tp = np.asarray(tp_flags, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = ctp / n_gt, ctp / (ctp + cfp)
    return float(compute_ap(recall, precision)[0])


# ---- predictions file -----------------------------------------------------------------------

PRED_DIR = V2_DATA / "predictions"


def prediction_files(names=None):
    """One CSV per model, written by 04_predict.py. v1_offshelf first, then the v2 runs sorted."""
    files = sorted(PRED_DIR.glob("*.csv"), key=lambda p: (p.stem != "v1_offshelf", p.stem))
    if names:
        files = [p for p in files if p.stem in set(names)]
    if not files:
        sys.exit(f"no prediction files in {PRED_DIR}; run v2/scripts/04_predict.py first")
    return files


def load_predictions(paths):
    """-> {(model, set, version): {tile: (xywhr (N,5), poly (N,4,2), conf (N,))}}, each tile's
    detections sorted by confidence, highest first. Tiles with no detection are absent."""
    import pandas as pd
    dt = {"model": str, "condition": str, "seed": str, "set": str, "version": str, "tile": str}
    df = pd.concat([pd.read_csv(p, dtype=dt, keep_default_na=False) for p in paths], ignore_index=True)
    out = {}
    for (m, s, v, t), g in df.groupby(["model", "set", "version", "tile"], sort=False):
        g = g.sort_values("conf", ascending=False, kind="stable")
        xywhr = g[["cx", "cy", "w", "h", "r"]].to_numpy(np.float32)
        poly = g[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].to_numpy(np.float32).reshape(-1, 4, 2)
        out.setdefault((m, s, v), {})[t] = (xywhr, poly, g["conf"].to_numpy(np.float32))
    return out


def models_in(paths):
    """[(model, condition, seed)] in file order, so every script iterates models identically.
    Taken from the file names (one file per model), so a model that predicted nothing still counts."""
    out = []
    for p in paths:
        if p.stem == "v1_offshelf":
            out.append((p.stem, "v1", ""))
        elif p.stem == "v1_offshelf_v1settings":
            out.append((p.stem, "v1_maxdet300", ""))
        else:
            cond, seed = p.stem.rsplit("_s", 1)
            out.append((p.stem, cond, seed))
    return out


def above(conf, c):
    """Ultralytics keeps a detection when conf > threshold (strictly), so v2 does too."""
    return np.asarray(conf) > c
