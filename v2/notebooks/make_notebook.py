"""Write notebooks/train.ipynb from the cells below, so the notebook stays readable in a diff.

    .venv/bin/python v2/notebooks/make_notebook.py
"""
import json
from pathlib import Path

INTRO = """# v2 training: ship-only YOLOv8n-OBB, three conditions x three seeds

Runs top to bottom on **Kaggle** (GPU T4, internet on) or **Colab** (one switch below).
Needs the private dataset made by `v2/scripts/03_package_for_kaggle.py` attached as input.

Set `RUN` and use *Save & Run All*:

1. `"timing"`: builds the training images, then trains 2 epochs of M_SR and writes a projected
   GPU budget. **Stop here and check the projection.**
2. `"seed0"`: M_HR, M_SR, M_BIC with seed 0. **Stop and sanity-check before spending more.**
3. `"seeds12"`: the same three conditions with seeds 1 and 2.

Each session rebuilds the bicubic and SR training images (about 10 minutes on a T4), because
`/tmp` does not survive between sessions. Download `v2_runs_<RUN>.zip` from the output tab and
unzip it into `v2/runs/` locally."""

SETTINGS = """PLATFORM = "kaggle"   # "kaggle" or "colab": the only switch
RUN = "timing"        # "timing" -> "seed0" -> "seeds12"   ("all" = seed0 + seeds12)
DEVICE = 0"""

PATHS = """import json, sys
from pathlib import Path

def find_upload(root):
    \"\"\"Locate code/v2 and data/ however the platform mounted the dataset; if it is still a zip,
    extract it once to /tmp/upload.\"\"\"
    root = Path(root)
    hits = list(root.glob("**/code/v2/kaggle_train.py"))
    if not hits:
        import zipfile
        zips = list(root.glob("**/*.zip"))
        assert zips, f"no dataset found under {root}: attach it with 'Add Input'"
        zipfile.ZipFile(zips[0]).extractall("/tmp/upload")
        root = Path("/tmp/upload")
        hits = list(root.glob("**/code/v2/kaggle_train.py"))
    data = next(root.glob("**/data/trainset_manifest.csv")).parent
    return hits[0].parent, data


if PLATFORM == "kaggle":
    CODE_V2, DATA = find_upload("/kaggle/input")
    OUT = Path("/kaggle/working")
elif PLATFORM == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    CODE_V2, DATA = find_upload("/content/drive/MyDrive/sr-ship-v2-trainset")  # the upload, copied to Drive
    OUT = Path("/content/drive/MyDrive/sr-ship-v2-output")
else:
    raise ValueError(PLATFORM)
WORK = Path("/tmp/v2work")                       # generated training images (not persisted)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
print("code:", CODE_V2, "\\ndata:", DATA, "\\noutput:", OUT)"""

INSTALL = """# pinned to the local environment; torch/torchvision come from the platform's CUDA build.
# A pin with no wheel for this platform falls back to the latest release and says so, rather than
# killing a background "Save & Run All"; the versions actually used are logged in the next cell.
import subprocess
for pin in ("ultralytics==8.4.171", "spandrel==0.4.2", "opencv-python==5.0.0.93"):
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", pin]).returncode:
        print(f"!! no {pin} for this platform; installing the latest {pin.split('==')[0]} instead")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pin.split("==")[0]], check=True)"""

ENV = """sys.path.insert(0, str(CODE_V2))
import kaggle_train as kt
print(json.dumps(kt._env(), indent=2))
!nvidia-smi --query-gpu=name,memory.total --format=csv"""

PREPARE = """rep_path = WORK / "prepare_report.json"
rep = json.loads(rep_path.read_text()) if rep_path.exists() else kt.prepare(DATA, WORK, device="cuda")
if rep["lr_hash_mismatch"]:
    print(f"NOTE: {rep['lr_hash_mismatch']} LR tiles differ bit-wise from the local ones "
          "(platform rounding); training images are still made with v1's functions.")
print("check tiles, PSNR of this GPU's SR vs local SR (dB):", rep["check_sr_psnr_vs_local_db"])"""

TRAIN = """PLAN = {
    "seed0": [(c, 0) for c in ("M_HR", "M_SR", "M_BIC")],
    "seeds12": [(c, s) for s in (1, 2) for c in ("M_HR", "M_SR", "M_BIC")],
}
PLAN["all"] = PLAN["seed0"] + PLAN["seeds12"]
manifest = DATA / "trainset_manifest.csv"

if RUN == "timing":
    kt.time_epochs(WORK, RUNS / "_timing", device=DEVICE)
else:
    for cond, seed in PLAN[RUN]:
        done = RUNS / f"{cond}_s{seed}" / "run_info.json"
        if done.exists():
            print(f"{cond}_s{seed} already finished, skipping")
            continue
        info = kt.train_one(cond, seed, WORK, RUNS, device=DEVICE, manifest=manifest)
        print(cond, seed, f"{info['seconds'] / 60:.1f} min", info["optimizer"], info["lr0_resolved"],
              "batch", info["batch_used"])"""

ZIP = """if RUN == "timing":
    kt.zip_runs(RUNS / "_timing", ["M_SR_s0"], OUT / "v2_runs_timing.zip", include_weights=False)
else:
    kt.zip_runs(RUNS, [f"{c}_s{s}" for c, s in PLAN[RUN]], OUT / f"v2_runs_{RUN}.zip")"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": [cell("markdown", INTRO)] + [cell("code", s) for s in
                                                (SETTINGS, PATHS, INSTALL, ENV, PREPARE, TRAIN, ZIP)]}
    out = Path(__file__).resolve().parent / "train.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
