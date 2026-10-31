"""Checkpoint 3: are the trained runs comparable, and did they learn? Run after unzipping a batch of
Kaggle runs into v2/runs/ (seed 0 first), before spending GPU time on the rest.

Per run and across runs:
  - batch actually used == batch requested (ultralytics silently shrinks it on out-of-memory)
  - epochs, warmup_epochs and close_mosaic are what config.yaml says for the condition
  - equal budget: epochs x batches-per-epoch within 2% across every run
  - optimizer and resolved lr0 identical across runs
  - no NaN in results.csv; final training losses below the first epoch's
  - last-epoch validation mAP50 above zero

Writes v2/results/checkpoint_runs.json and v2/results/figures/training_curves.png: losses and
validation mAP50 against the fraction of training done, because the conditions run different epoch
counts over the same number of steps. Exits non-zero if any check fails.
"""
import argparse, csv, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2, V2_RESULTS, CONDITION_COLOURS  # noqa: E402

T = CFG["train"]
CURVES = [("train/box_loss", "train box loss"), ("train/cls_loss", "train class loss"),
          ("metrics/mAP50(B)", "validation mAP50")]
BUDGET_TOL = 0.02


def load_runs(runs_dir):
    runs = []
    for d in sorted(Path(runs_dir).glob("M_*_s*")):
        info_f, res_f = d / "run_info.json", d / "results.csv"
        if not (info_f.exists() and res_f.exists()):
            continue
        with open(res_f) as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        runs.append({"name": d.name, "info": json.loads(info_f.read_text()), "rows": rows})
    return runs


def check_run(r):
    i, rows, a = r["info"], r["rows"], r["info"]["train_args"]
    c = T["conditions"][i["condition"]]
    fails = []
    if i["batch_used"] != i["batch_requested"]:
        fails.append(f"batch reduced {i['batch_requested']} -> {i['batch_used']} (out of memory)")
    for k in ("epochs", "warmup_epochs", "close_mosaic"):
        if a[k] != c[k]:
            fails.append(f"{k} {a[k]} but config says {c[k]}")
    nums = [float(v) for row in rows for k, v in row.items() if k != "epoch" and v.strip()]
    if any(math.isnan(x) for x in nums):
        fails.append("NaN in results.csv")
    for k in ("train/box_loss", "train/cls_loss"):
        if float(rows[-1][k]) >= float(rows[0][k]):
            fails.append(f"{k} did not fall ({float(rows[0][k]):.3f} -> {float(rows[-1][k]):.3f})")
    if float(rows[-1]["metrics/mAP50(B)"]) <= 0:
        fails.append("last-epoch validation mAP50 is 0")
    return {"run": r["name"], "condition": i["condition"], "seed": i["seed"], "epochs": a["epochs"],
            "n_train_images": i["n_train_images"], "batches_per_epoch": i["batches_per_epoch"],
            "total_batches": a["epochs"] * i["batches_per_epoch"], "batch_used": i["batch_used"],
            "optimizer": i["optimizer"], "lr0": i["lr0_resolved"], "minutes": round(i["seconds"] / 60, 1),
            "last_val_map50": float(rows[-1]["metrics/mAP50(B)"]), "gpu": i["env"].get("gpu", ""),
            "fails": fails}


def check_across(rows):
    fails = []
    budgets = [r["total_batches"] for r in rows]
    if max(budgets) > (1 + BUDGET_TOL) * min(budgets):
        fails.append(f"unequal budgets: {dict((r['run'], r['total_batches']) for r in rows)}")
    if len({(r["optimizer"], r["lr0"]) for r in rows}) > 1:
        fails.append(f"optimizer/lr0 differ: {dict((r['run'], (r['optimizer'], r['lr0'])) for r in rows)}")
    return fails


def dodge_labels(ax, ends, ink, gap=0.08):
    """Direct-label each condition at its line end, pushed apart vertically so converging curves
    never print labels on top of each other (gap = fraction of the y range)."""
    lo, hi = ax.get_ylim()
    step = gap * (hi - lo)
    placed = []
    for cond, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_lab = max(y, placed[-1] + step) if placed else y
        placed.append(y_lab)
        ax.annotate(cond, (x, y), xytext=(x + 0.015, y_lab), textcoords="data", va="center",
                    fontsize=8, color=ink)
    ax.set_ylim(lo, max(hi, placed[-1] + step / 2))


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#2b2b2a", "#6b6a64", "#e4e3de", "#fcfcfb"
    order = list(T["conditions"])                       # fixed condition order, never alphabetical
    runs = sorted(runs, key=lambda r: (order.index(r["info"]["condition"]), r["info"]["seed"]))
    fig, axes = plt.subplots(1, len(CURVES), figsize=(13, 3.8), facecolor=surface)
    for ax, (key, title) in zip(axes, CURVES):
        ax.set_facecolor(surface)
        ends = {}
        for r in runs:
            cond, n = r["info"]["condition"], len(r["rows"])
            x = [(k + 1) / n for k in range(n)]
            y = [float(row[key]) for row in r["rows"]]
            ax.plot(x, y, color=CONDITION_COLOURS[cond], lw=2 if r["info"]["seed"] == 0 else 1.2,
                    label=cond if cond not in ends else None)
            ends.setdefault(cond, (x[-1], y[-1]))
        dodge_labels(ax, ends, ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("fraction of training done", fontsize=8, color=muted)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(0, 1.12)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=surface)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(V2 / "runs"))
    ap.add_argument("--out", default=str(V2_RESULTS), help="where the json and figure go (tests use a temp dir)")
    args = ap.parse_args()
    out = Path(args.out)
    runs = load_runs(args.runs)
    if not runs:
        sys.exit(f"no finished runs (run_info.json + results.csv) under {args.runs}")
    rows = [check_run(r) for r in runs]
    across = check_across(rows)
    print(f"{'run':10s} {'epochs':>6s} {'imgs':>5s} {'b/ep':>5s} {'budget':>7s} {'batch':>5s} "
          f"{'opt':>6s} {'lr0':>7s} {'min':>6s} {'val mAP50':>9s}  checks")
    for r in rows:
        print(f"{r['run']:10s} {r['epochs']:6d} {r['n_train_images']:5d} {r['batches_per_epoch']:5d} "
              f"{r['total_batches']:7d} {r['batch_used']:5d} {r['optimizer']:>6s} {r['lr0']:7.4f} "
              f"{r['minutes']:6.1f} {r['last_val_map50']:9.3f}  {'ok' if not r['fails'] else '; '.join(r['fails'])}")
    for f in across:
        print("ACROSS RUNS: " + f)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint_runs.json").write_text(json.dumps({"runs": rows, "across_runs_fails": across}, indent=2))
    plot(runs, out / "figures" / "training_curves.png")
    print(f"-> {out / 'checkpoint_runs.json'}\n-> {out / 'figures' / 'training_curves.png'}")
    if across or any(r["fails"] for r in rows):
        sys.exit(1)
    print("all checks passed: next, 04_predict.py, 05_eval_setA.py, 06_eval_setB_firing.py")


if __name__ == "__main__":
    main()
