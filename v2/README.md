# v2: does training the detector on super-resolved images remove the invented ships?

v1 put Real-ESRGAN in front of an off-the-shelf ship detector that had never seen super-resolved
imagery. A reviewer pointed out that this mixes two effects: the SR model inventing detail, and a
detector meeting SR artefacts for the first time. v2 separates them by training small ship-only
detectors three ways and testing each on the same tiles as v1.

| Condition | Trained on | Question it answers |
|---|---|---|
| `M_HR` | original tiles | baseline: never seen a restored image |
| `M_SR` | original + Real-ESRGAN versions of the same tiles | the reviewer's suggestion |
| `M_BIC` | original + bicubic versions of the same tiles | control: is it SR, or any restored image? |

- **Q1:** does training on SR reduce false ships and improve recall on SR inputs? (`M_SR` vs `M_HR` on SR)
- **Q2:** with each detector trained on what it is tested on, does the SR pipeline still invent more
  vessels than the bicubic one? (`M_SR` on SR vs `M_BIC` on bicubic)

The plan, including every pre-registered comparison, is in [`PLAN.md`](PLAN.md). The results, the
verdict on both questions and the limitations are in [`results/summary.md`](results/summary.md),
which `scripts/11_summary.py` generates from the results files.

## Phase 0 findings (already done)

[`results/phase0_sr_audit.md`](results/phase0_sr_audit.md) checks v1's SR pipeline for bugs.

- **SR implementation: no bug.** v1's output is bit-identical to the official Real-ESRGAN inference
  code on the same device, and v1's degraded and bicubic images recompute bit-for-bit.
- **Greyscale tiles come out coloured.** That is the model, not the code. Converting them back to grey
  ("v1b") barely moves v1's result.
- **v1's Set A numbers were truncated.** v1 called the detector with ultralytics' default
  `max_det=300`; two dense harbour tiles hit that cap, which removed 123 ship detections (89 of
  them correct) on the original images at conf 0.25, and 92–119 on each restored version. Corrected
  recall is 2.4–2.7 points higher for every version; the SR-minus-bicubic recall gap at conf 0.25
  moves from −0.020 to −0.022. See section 3 of the Phase 0 report and `results/setA_metrics.json`
  (`_v1_max_det_truncation`). v2 predicts with `max_det=1000` and records any image that hits it.

## Reproducing v2

Everything runs from the repository root with v1's environment (`.venv`, Python 3.11; versions
pinned in [`requirements.txt`](requirements.txt)). DOTA-v1.0 must already be in `data/DOTAv1/`
(the ultralytics layout v1 used). DOTA is licensed for academic, non-commercial use, so no DOTA
imagery or derived tiles are in this repository.

### 0. Audit v1 (local)

The official reference needs `realesrgan` 0.3.0 and `basicsr` 1.4.2 in a separate, throwaway
environment. basicsr 1.4.2 fails to build its wheel on recent setuptools and imports a torchvision
module that no longer exists, so use its source tree directly:

```bash
python3.11 -m venv /tmp/refvenv
echo "$PWD/.venv/lib/python3.11/site-packages" > /tmp/refvenv/lib/python3.11/site-packages/v1.pth
/tmp/refvenv/bin/pip download --no-deps --no-binary :all: basicsr==1.4.2 -d /tmp/dl
tar xzf /tmp/dl/basicsr-1.4.2.tar.gz -C /tmp/dl
echo /tmp/dl/basicsr-1.4.2 > /tmp/refvenv/lib/python3.11/site-packages/basicsr.pth
/tmp/refvenv/bin/pip install --no-deps realesrgan==0.3.0 addict yapf lmdb future
sed -i '' 's/functional_tensor import/functional import/' /tmp/dl/basicsr-1.4.2/basicsr/data/degradations.py

.venv/bin/python v2/scripts/00_sr_audit.py --ref-python /tmp/refvenv/bin/python
.venv/bin/python v2/scripts/00b_v1b_grey.py
```

### 1–2. Training set (local)

```bash
.venv/bin/python v2/scripts/01_build_trainset.py      # 600 ship / 300 no-ship tiles from DOTA train
.venv/bin/python v2/scripts/02_degrade_train.py       # LR hashes + 5 cross-device check tiles
.venv/bin/python v2/scripts/03_package_for_kaggle.py  # v2/kaggle_upload/, prints its size
```

### 3. Training (Kaggle GPU, Colab fallback)

1. Upload `v2/kaggle_upload/` as a **private** Kaggle dataset (DOTA licence), e.g.
   `kaggle datasets create -p v2/kaggle_upload` after putting your username in
   `dataset-metadata.json`.
2. Build the notebook: `.venv/bin/python v2/notebooks/make_notebook.py`, import
   `v2/notebooks/train.ipynb` into Kaggle, attach the dataset, accelerator GPU T4, internet on.
3. Set `RUN = "timing"` and *Save & Run All*. Read the projected GPU hours in the output.
4. `RUN = "seed0"`, then `RUN = "seeds12"`. Download each `v2_runs_<RUN>.zip` and unzip into `v2/runs/`.

On Colab, copy `v2/kaggle_upload/` to Drive as `MyDrive/sr-ship-v2-trainset` and set
`PLATFORM = "colab"`.

### 4–7. Evaluation, audit, statistics (local)

```bash
.venv/bin/python v2/scripts/checkpoint3.py            # runs comparable? learned? + training curves
.venv/bin/python v2/scripts/04_predict.py             # every model in v2/runs/ + the v1 detector
.venv/bin/python v2/scripts/05_eval_setA.py           # exits non-zero if v1's numbers don't reproduce
.venv/bin/python v2/scripts/06_eval_setB_firing.py    # same gate for Set B
.venv/bin/python v2/scripts/07_build_audit_crops.py   # prints how many crops need a human verdict
.venv/bin/python v2/scripts/08_label.py               # blind labelling (s / n / u)
.venv/bin/python v2/scripts/09_recount_invention.py  # refuses while a compared cell is unlabelled
.venv/bin/python v2/scripts/10_stats.py
.venv/bin/python v2/scripts/11_summary.py             # results/summary.md + figures
```

`08_label.py` lists the crops the comparisons need first (1–767 of 1,218); the rest are optional,
and any cell with an unlabelled crop is reported as not audited rather than counted partially.

Unit tests: `python -m pytest v2/tests -q` (needs pytest; any environment with v1's packages).

## Layout

```
v2/
  PLAN.md  README.md  config.yaml  requirements.txt
  common_v2.py        config, read-only imports of v1 code, rotated-box matching, AP
  kaggle_train.py     builds the three training sets on the GPU and trains one condition x seed
  scripts/            00-11, one per phase
  notebooks/          make_notebook.py -> train.ipynb
  tests/              unit tests
  results/            everything the write-up quotes
  data/  runs/        local only (gitignored)
```
