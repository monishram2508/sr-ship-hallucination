"""Shared config for the SR-hallucination harness."""
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SR_WEIGHTS = WEIGHTS / "RealESRGAN_x4plus.pth"
DETECTOR = "yolov8s-obb.pt"

TILE = 1024          # HR tile size
SCALE = 4            # SR factor
LR = TILE // SCALE   # 256
BLUR_SIGMA = 0.5 * SCALE   # Shermeyer & Van Etten: 0.5 * GSD_out / GSD_native
IMGSZ = 1024
SHIP_CLS = 1
HARBOR_CLS = 7
BRIDGE_CLS = 8


def device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---- statistics helpers (shared by 06_summary.py and 09_recount.py) ----------

def bootstrap_pct_ci(hits, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for '% of tiles with >=1 invented ship'.
    hits: 0/1 array, one entry per tile. Resamples TILES, which is the unit of independence."""
    import numpy as np
    hits = np.asarray(hits, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    stat = hits[idx].mean(1) * 100
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_diff_ci(hits_a, hits_b, n_boot=10000, seed=0, alpha=0.05):
    """Paired bootstrap CI for (pct_a - pct_b). The SAME resampled tile indices are used for both
    arms, because the two versions are measured on the same tiles."""
    import numpy as np
    a, b = np.asarray(hits_a, float), np.asarray(hits_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    stat = (a[idx].mean(1) - b[idx].mean(1)) * 100
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(hits_a, hits_b):
    """Exact (binomial) McNemar for paired binary outcomes on the same tiles.
    Returns (b, c, p) where b = a-only hits, c = b-only hits. Exact, not chi-square:
    the discordant counts here are small and one cell is often 0."""
    import numpy as np
    from scipy.stats import binomtest
    a, b_ = np.asarray(hits_a, int), np.asarray(hits_b, int)
    b = int(((a == 1) & (b_ == 0)).sum())
    c = int(((a == 0) & (b_ == 1)).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
