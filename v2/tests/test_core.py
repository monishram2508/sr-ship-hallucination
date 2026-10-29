"""Unit tests for the pieces of v2 most likely to fail silently.

    .venv/bin/python -m pytest v2/tests -q
"""
import importlib.util, sys
from pathlib import Path

import numpy as np
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import common_v2 as c2  # noqa: E402


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem.replace("-", "_"), V2 / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def box(cx, cy, w=40, h=12, r=0.0):
    return [cx, cy, w, h, r]


def test_greedy_tp_matches_and_rejects():
    gt = np.array([box(100, 100), box(300, 300)], np.float32)
    det = np.array([box(101, 100), box(600, 600), box(300, 301)], np.float32)
    assert c2.greedy_tp(gt, det, 0.5).tolist() == [True, False, True]


def test_greedy_tp_one_gt_matched_once():
    gt = np.array([box(100, 100)], np.float32)
    det = np.array([box(100, 100), box(101, 100)], np.float32)   # duplicate: second is a FP
    assert c2.greedy_tp(gt, det, 0.5).tolist() == [True, False]


def test_greedy_tp_prefix_property():
    """Counts at a higher threshold must equal a prefix of the single pass at 0.01."""
    rng = np.random.default_rng(0)
    gt = np.array([box(*rng.uniform(50, 950, 2)) for _ in range(20)], np.float32)
    det = np.concatenate([gt + rng.normal(0, 3, gt.shape).astype(np.float32),
                          np.array([box(*rng.uniform(50, 950, 2)) for _ in range(15)], np.float32)])
    det = det[rng.permutation(len(det))]
    full = c2.greedy_tp(gt, det, 0.5)
    for k in (1, 5, 17, 30):
        assert c2.greedy_tp(gt, det[:k], 0.5).tolist() == full[:k].tolist()


def test_ap50_extremes():
    assert c2.ap50([1, 1, 1], [0.9, 0.8, 0.7], 3) == pytest.approx(1.0, abs=0.01)
    assert c2.ap50([0, 0], [0.9, 0.8], 3) == 0.0
    assert c2.ap50([], [], 0) == 0.0


def test_above_is_strict_like_ultralytics():
    assert c2.above([0.25, 0.2500001, 0.5], 0.25).tolist() == [False, True, True]


def test_kappa():
    k = load("scripts/09_recount_invention.py").kappa
    assert k(["ship", "not_ship"], ["ship", "not_ship"]) == 1.0
    # 2x2 textbook case: po = 0.7, pe = 0.5 -> kappa 0.4
    a = ["y"] * 5 + ["n"] * 5
    b = ["y"] * 4 + ["n"] + ["y"] * 2 + ["n"] * 3
    assert k(a, b) == pytest.approx(0.4)
    assert k([], []) is None


def test_ship_only_label_clips_and_drops(tmp_path):
    m01 = load("scripts/01_build_trainset.py")
    f = tmp_path / "t.txt"
    f.write_text("1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
                 "1 0.9 0.9 1.08 0.9 1.08 1.0 0.9 1.0\n"
                 "9 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")
    lines, clipped = m01.ship_only_label(f)
    assert clipped == 1 and len(lines) == 2
    assert all(l.startswith("0 ") for l in lines)
    assert max(float(x) for l in lines for x in l.split()[1:]) <= 1.0
    assert m01.ship_only_label(tmp_path / "missing.txt") == ([], 0)


def test_equal_training_budget_in_config():
    """Every condition sees the same number of images x epochs, and warmup / close_mosaic scale with
    epochs, so all three get the same schedule measured in steps."""
    conds = c2.CFG["train"]["conditions"]
    steps = {k: c["epochs"] * len(c["sources"]) for k, c in conds.items()}
    assert len(set(steps.values())) == 1
    for c in conds.values():
        assert c["warmup_epochs"] * len(c["sources"]) == 3.0
        assert c["close_mosaic"] * len(c["sources"]) == 10


def test_train_args_use_condition_settings():
    kt = load("kaggle_train.py")
    a = kt._train_args("M_SR", 1, "d.yaml", "out", "M_SR_s1", 0)
    assert (a["epochs"], a["warmup_epochs"], a["close_mosaic"], a["seed"]) == (50, 1.5, 5, 1)
    assert a["single_cls"] and a["deterministic"] and a["patience"] == 0 and a["flipud"] == 0.5
    t = kt._train_args("M_SR", 0, "d.yaml", "out", "x", 0, epochs=2, close_mosaic=0)
    assert (t["epochs"], t["close_mosaic"]) == (2, 0)


def test_binary_comparison_pairs_by_seed():
    m10 = load("scripts/10_stats.py")
    n = 50
    arrs = {}
    for s in ("0", "1", "2"):
        a = np.zeros(n, int)
        a[:10] = 1                       # A fires on 10 tiles
        b = np.zeros(n, int)
        b[:2 + int(s)] = 1               # B fires on 2-4 tiles, all shared with A
        arrs[("M_SR", "sr", s)], arrs[("M_BIC", "bicubic", s)] = a, b
    r = m10.binary_comparison(arrs, ("M_SR", "sr"), ("M_BIC", "bicubic"), ["0", "1", "2"])
    assert r["n_seeds"] == 3 and r["same_sign_all_seeds"]
    assert r["per_seed"]["0"]["a_only_tiles"] == 8 and r["per_seed"]["0"]["b_only_tiles"] == 0
    assert r["diff_mean"] == pytest.approx(100 * (10 - 3) / n)
    lo, hi = r["seed_avg_diff_ci95"]
    assert lo <= r["diff_mean"] <= hi
