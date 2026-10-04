# Does GAN super-resolution make detectors see ships that aren't there?

An object-level test of super-resolution as a pre-processing step. A GAN super-resolver is placed
in front of an off-the-shelf ship detector that was never retrained on its output, and the two
questions are counted separately: how often does it **recover** objects lost to resolution, and how
often does it **invent** objects that were never there?

Short answer, on 150 harbor and bridge tiles where the detector saw nothing in the original image:
**super-resolution made the detector claim vessels that are not there on 14-16% of tiles, against
1.3-2.0% for a plain bicubic upscale of the same degraded data** (exact McNemar p = 3.8e-06).
It recovered nothing: ship recall at a matched threshold was 0.881 against bicubic's 0.901.
PSNR saw none of it.

![Hero figure: four panels showing the original, the degraded image, a bicubic upscale and the
Real-ESRGAN output, where only the last produces ship detections](results/figures/hero.png)

---

## The problem

Super-resolution is increasingly bolted onto the front of imagery pipelines that were built and
tuned on real captures: sharpen the frame, then run the detectors, trackers and reports that already
exist. The sharpening model is usually validated with pixel-similarity scores (PSNR, SSIM) or with a
radiometric check that compares the output's brightness back against the input. Neither of those can
see objects. A generative model that paints a plausible hull-and-shadow onto open water can leave
average brightness untouched at the input scale and still hand the detector a vessel; everything
downstream — counting, cross-referencing, alerting — inherits that object as if a sensor had seen it.
This repository measures that failure directly, at the level of objects rather than pixels, and
splits the result into recovery and invention instead of reporting one aggregate score.

The information argument is the reason the measurement is possible at all. The super-resolved image
is computed from a 4x degraded copy of the original, so by the data processing inequality it cannot
contain more information about the scene than the original does. A detection that appears on the
super-resolved image and *not* on the original is therefore never new evidence. It is either
synthesized texture, or the same evidence re-formatted into edges and contrast the detector's priors
accept. Which one it is has to be decided by looking — hence the hand audit below.

## Results

Detector: YOLOv8s-OBB pretrained on DOTAv1, never retrained or fine-tuned on super-resolved imagery.
Super-resolver: Real-ESRGAN x4plus. Baseline: bicubic x4 of the identical degraded image.

### Recovery — Set A, 100 tiles carrying 3380 labelled ship instances

Ship class only, same fixed confidence for every version, greedy highest-confidence-first matching,
rotated-box IoU >= 0.5.

| conf | Version | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|---|
| 0.25 | Original | 0.890 | **0.951** | 0.920 | 0.982 |
| 0.25 | Bicubic x4 | 0.874 | **0.901** | 0.887 | 0.946 |
| 0.25 | Real-ESRGAN x4 | 0.845 | **0.881** | 0.863 | 0.934 |
| 0.50 | Original | 0.913 | 0.935 | 0.924 | — |
| 0.50 | Bicubic x4 | 0.928 | 0.853 | 0.889 | — |
| 0.50 | Real-ESRGAN x4 | 0.916 | 0.825 | 0.868 | — |

**The GAN did not recover anything.** It sits 2.0 points of recall below a plain bicubic upscale at
conf 0.25 and 2.8 points below at conf 0.50. Both are far below the original. This contradicts
hypothesis H1 and is consistent with the 2026 GeoSR-Bench finding that GAN and diffusion
super-resolvers can do worse on a downstream task than not upscaling at all.

### Invention — Set B, 150 tiles where the detector saw nothing in the original

Set B is harbor and bridge tiles with **zero ship labels and zero ship detections on the original**
at conf 0.25. That construction is weaker than it looks and the audit is what makes the numbers
usable: it verifies zero *detections*, not zero vessels. The dataset omits small craft, so some of
these tiles do contain boats nobody labelled and the detector missed on the original. Every extra
box was therefore judged by hand against the original imagery (protocol below), and recoveries of
real unlabelled vessels are separated from inventions rather than counted as inventions.

**Tile-level, not box-level**: each tile contributes at most one event, so no single pathological
scene can drive the rate. Confidence intervals are percentile bootstrap over 10,000 resamples of the
150 tiles (seed 0). Significance is the exact binomial McNemar test on the paired tiles.

**(1) The firing rate — needs no human verdict at all.** How often did degrading and restoring make
the detector fire where the sharp original was silent?

| conf | Version | tiles | % of tiles | 95% CI | paired difference | exact McNemar p |
|---|---|---|---|---|---|---|
| 0.25 | Real-ESRGAN x4 | 43/150 | **28.7%** | [21.3, 36.0] | +22.7 pp [16.0, 29.3] | **1.2e-10** |
| 0.25 | Bicubic x4 | 9/150 | 6.0% | [2.7, 10.0] | — | — |
| 0.50 | Real-ESRGAN x4 | 14/150 | **9.3%** | [4.7, 14.0] | +8.0 pp [3.3, 13.3] | **4.2e-03** |
| 0.50 | Bicubic x4 | 2/150 | 1.3% | [0.0, 3.3] | — | — |

**(2) Invention — tiles carrying at least one box a blind audit judged *not a vessel*.** The lower
band counts only boxes judged not-a-vessel; the upper band also counts the handful still uncertain
after two passes.

| conf | band | Real-ESRGAN | Bicubic | difference | exact McNemar p |
|---|---|---|---|---|---|
| 0.25 | lower | **14.0%** [8.7, 20.0] | 1.3% [0.0, 3.3] | +12.7 pp | **3.8e-06** |
| 0.25 | upper | **16.0%** [10.0, 22.0] | 2.0% [0.0, 4.7] | +14.0 pp | **5.7e-06** |
| 0.50 | lower | 2.7% [0.7, 5.3] | 0.7% [0.0, 2.0] | +2.0 pp | 0.375 |
| 0.50 | upper | 3.3% [0.7, 6.7] | 0.7% [0.0, 2.0] | +2.7 pp | 0.219 |

**Invention is significant at conf 0.25 and not at conf 0.50.** Only 31 super-resolved boxes survive
the higher threshold at all, 12-13 of them not-a-vessel on 4-5 tiles — too little data to resolve
the difference, not evidence that it disappears. The point estimate still moves the same way, and
the verdict-free firing rate *is* significant at that threshold (p = 4.2e-03). Raising the
detector's confidence bar reduces invention; it does not remove it.

**One tile dominates, and the result survives removing it.** A single dark residential hillside
carries 45 of the 85 not-a-vessel boxes in the whole set. Dropping that tile from the numerator
*and* the denominator leaves 149 tiles and barely moves the tile-level result:

| conf 0.25 band | all 150 tiles | worst tile dropped (149) | bicubic | exact McNemar p |
|---|---|---|---|---|
| lower | 14.0% | **13.4%** [8.1, 19.5] | 1.3% | **7.6e-06** |
| upper | 16.0% | **15.4%** [10.1, 21.5] | 2.0% | **1.1e-05** |

**(3) Box-level composition — the two arms are different kinds of error.** This figure *does* depend
on that one tile, so both versions are always given together.

| extra boxes where the original was silent (conf 0.25) | total | real vessel | no vessel | uncertain |
|---|---|---|---|---|
| Real-ESRGAN x4 | 140 | 52 | 83 | 5 |
| Real-ESRGAN x4, worst tile dropped | 95 | 52 | 38 | 5 |
| Bicubic x4 | 13 | 10 | 2 | 1 |

59-63% of the GAN's extra boxes had no vessel under them set-wide, 40-45% with the worst tile
dropped. Bicubic's 13 extra boxes contained 2-3 with no vessel — 13 boxes is too few for a
percentage to carry meaning, so it is given as a count. The rest of bicubic's extra boxes (10 of 13)
were real unlabelled vessels. **When a plain upscale makes the detector newly confident it is
usually right; when the GAN does it, most of the time there is nothing there.**

### What the pixel metrics said

| Version | mean PSNR vs original | mean SSIM |
|---|---|---|
| Bicubic x4 | 30.12 dB | 0.753 |
| Real-ESRGAN x4 | 28.78 dB | 0.748 |

A 1.3 dB gap, while the tile-level invention rates differ by roughly 10x. The pixel scores happen to
rank bicubic first, but nothing in the number distinguishes "slightly less faithful texture" from
"draws vessels that do not exist". Within the super-resolved arm, PSNR does not even rank the tiles:

![Scatter plot of per-tile PSNR against audited invented objects, showing no
relationship](results/figures/psnr_vs_detection_delta.png)

Low PSNR predicts a tile carrying an invented object with AUC 0.59 (p = 0.17), Spearman rho = -0.11
(p = 0.17) — indistinguishable from a coin flip. Before the audit the same plot looked predictive
(AUC 0.68, p = 0.0005); that signal was real unlabelled vessels, not inventions, and it came through
a confound: PSNR tracks how busy a scene is (Spearman rho = -0.44 against labelled object count,
p = 2e-08), and busy scenes are where the unlabelled boats live. **Measuring invention without
auditing the boxes reproduces that mistake.**

## Method

```
                       Gaussian PSF blur            INTER_AREA
  original 1024px  ----  sigma = 2.0  ---->  decimate 4x  ---->  LR 256px
  (ground truth)                                                    |
                                                                    |-- bicubic x4 --> 1024px
                                                                    |                  (baseline, no AI)
                                                                    |
                                                                    '-- Real-ESRGAN --> 1024px
                                                                        x4plus          (super-resolved)

  original / bicubic / super-resolved  ---->  SAME pretrained ship detector  ---->  compare
                                              (YOLOv8s-OBB, never retrained)         against the
                                                                                     SAME labels
```

The degradation follows Shermeyer & Van Etten (2019): a Gaussian point-spread-function blur with
sigma = 0.5 x GSD_out / GSD_native (2.0 at 4x), then inter-area decimation. The blur models the
telescope optics smearing each point of light before pixels exist; decimation models the coarser
pixel grid. Everything is saved as PNG — JPEG artefacts would confound the comparison.

Two test sets, both tiled 1024x1024 from the validation split only:

- **Set A (recovery)** — 100 random tiles with at least one ship label, seed 0, 3380 ship instances.
- **Set B (invention)** — 150 tiles with a harbor or bridge label, zero ship labels, and zero ship
  detections on the original at conf 0.25.

| script | what it does |
|---|---|
| `01_tile.py` | tile the val split to 1024x1024 |
| `02_select_sets.py` | build Set A and Set B |
| `03_degrade_sr.py` | blur, decimate, bicubic, Real-ESRGAN; write the three versions |
| `04_detect.py` | run the detector on every version of both sets |
| `05_psnr.py` | PSNR / SSIM per tile |
| `08_audit_boxes.py` | build the blinded audit: crops, key, empty label file |
| `11_label.py` | the blind labelling UI |
| `13_pass2.py`, `14_merge.py` | second-pass adjudication of the uncertain crops, and the merge |
| `09_recount.py` | recount Set B from the audit verdicts |
| `10_fixed_threshold.py` | Set A P/R/F1 at a *fixed* confidence for all versions |
| `06_summary.py`, `16_headline.py` | build `results/summary.md` and `results/headline.md` from files |
| `07_figures.py` | hero figure, 3-panel comparison, PSNR scatter |

## The audit protocol

The invention numbers rest on human verdicts, so the protocol is designed to make those verdicts
hard to bend.

1. **Every extra box is audited, both arms.** All 153 ship boxes that either version produced on
   Set B at conf 0.25 — super-resolved and bicubic alike — go into the same pool. Auditing only the
   super-resolved arm would guarantee the asymmetry it is trying to measure.
2. **Crops come from the original image, never from the version that produced the box.** The
   question is whether a vessel is there, and the original is the best available evidence of that.
3. **Blind.** The pool is split into `audit_key.csv` (crop id -> version, tile, confidence, box) and
   `audit_labels.csv` (crop id -> verdict). Crop order is shuffled with a fixed seed *before* ids are
   assigned, so the ids leak nothing — no version, no tile, no confidence. The labelling UI reads
   only the labels file and never opens the key.
4. **Three verdicts, and blank is a hard error.** `ship` / `not_ship` / `unsure`. The recount script
   refuses to run if any crop is unlabelled, rather than defaulting a blank to anything — a silent
   default would manufacture a result.
5. **Two bands instead of a judgement call.** Lower bound counts only `not_ship`; upper bound also
   counts `unsure`. Every invention number is reported as that range.
6. **Two passes.** Pass 1 used 48 px of context around each box; the 38 crops labelled `unsure` were
   re-rendered with 128 px, reshuffled under a new seed, and judged again, still blind. They resolved
   25 to `not_ship`, 7 to `ship`, 6 still uncertain. Pass 1 is preserved at
   `results/audit_labels_pass1.csv` and both passes are reported side by side in
   `results/summary.md`. The second pass narrowed the invention band from 6.0-18.0% to 14.0-16.0%.
7. **The recount is verified against the raw counts.** Forcing every verdict to `not_ship`
   reproduces the pre-audit detection counts exactly, per tile, with zero mismatches — so the join
   between key, labels and counts is sound.

Contact sheets of every verdict are in `results/figures/audit/` so the labelling can be inspected
rather than trusted:

![Contact sheet of every crop judged not a vessel, magenta box on each claimed
detection](results/figures/audit/qa_not_ship.png)

## Related work

- **Shermeyer & Van Etten (2019), [arXiv:1812.04098](https://arxiv.org/abs/1812.04098).** Super-
  resolution applied to satellite imagery for object detection, on 30 cm imagery with VDSR and a
  random-forest super-resolver. Their detectors were **retrained on super-resolved output**, which is
  the opposite of the setting here: this project asks what happens when SR is bolted in front of a
  detector nobody retrained. Their sensor-degradation model is the one used here.
- **GeoSR-Bench, Li et al. (2026), [arXiv:2605.00310](https://arxiv.org/abs/2605.00310).** Nine
  super-resolution models across transformer, neural-operator, GAN and diffusion families, evaluated
  on downstream geospatial tasks. Finding: PSNR and SSIM often fail to track downstream performance
  and sometimes correlate negatively with it; in places ESRGAN and a diffusion model did worse than
  no upscaling at all. Their downstream tasks are **pixel-level** (segmentation, regression) — there
  is no object detection, and so no count of invented objects.

**The gap this fills:** a direct count of invented *objects* in scenes verified empty of detections,
using GAN super-resolution with a detector that was not retrained, with every claimed object
adjudicated by a blind hand audit rather than assumed from the label file.

## Limitations

1. **The degradation is simulated.** A Gaussian PSF blur plus decimation is not a real coarser
   sensor: no atmosphere, no real optics, no sensor noise model, no compression.
2. **One super-resolver, one detector.** Real-ESRGAN x4plus and YOLOv8s-OBB. The result is about
   this pair; it is evidence about GAN super-resolution as a class, not proof about any other model.
   Real-ESRGAN is also a GAN trained on everyday photographs, not a satellite-trained model.
3. **The detector may have seen these images in training.** It is pretrained on the same dataset
   whose validation split is used here, so the original-image scores are an upper bound. The
   *relative* comparison between the three versions is unaffected, since all three are scored with
   the same detector against the same labels.
4. **Single labeller, and not blind to the hypothesis.** The audit was done by the author. Verdicts
   were blind to version, tile and confidence, but not to what the project is testing. Two
   mitigations are in place and neither is a substitute for a second labeller: both arms were audited
   in the same blind pool, and every number is reported as a band rather than a point. The direction
   of the second pass is worth stating plainly — with more context the uncertain crops resolved
   25 to not-a-vessel against 7 to vessel, which moved the lower band *up*.
5. **Invention at conf 0.50 is underpowered.** 31 surviving boxes on 150 tiles cannot resolve a
   2-point difference. That cell is reported as non-significant, not as null.
6. **Boxes cluster within tiles**, so boxes are not independent observations. Every significance test
   here is therefore tile-level, where the unit of independence is the tile; box-level figures are
   reported as descriptive composition only, always alongside the version with the dominant tile
   removed.
7. **The source dataset's labels are incomplete.** It omits small craft, which is exactly why the
   hand audit exists and why "unmatched detection" is used instead of "false positive" throughout:
   the original imagery itself leaves 397 ship detections unmatched against the same labels.
8. **The dataset mixes aerial and satellite imagery** at varying ground sample distances, so "4x
   degradation" is not a single physical scale.

## Reproducing

```bash
python3.11 -m venv .venv
.venv/bin/pip install ultralytics spandrel opencv-python scikit-image pandas matplotlib tqdm scipy

# data (academic use only, see below) and super-resolution weights
curl -L -o data/DOTAv1.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip
curl -L -o weights/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

.venv/bin/python scripts/00_smoke.py          # environment check
.venv/bin/python scripts/01_tile.py
.venv/bin/python scripts/02_select_sets.py
.venv/bin/python scripts/03_degrade_sr.py
.venv/bin/python scripts/04_detect.py
.venv/bin/python scripts/05_psnr.py
.venv/bin/python scripts/10_fixed_threshold.py

# the audit
.venv/bin/python scripts/08_audit_boxes.py    # builds crops + key + blank label file
.venv/bin/python scripts/11_label.py          # s = ship, n = not_ship, u = unsure, b = back, q = quit
.venv/bin/python scripts/13_pass2.py
.venv/bin/python scripts/11_label.py --pass2
.venv/bin/python scripts/14_merge.py

# results
.venv/bin/python scripts/09_recount.py
.venv/bin/python scripts/09_recount.py --exclude P1751__1024__0___2976 --out results/setB_audited_no_P1751.json
.venv/bin/python scripts/06_summary.py
.venv/bin/python scripts/16_headline.py
.venv/bin/python scripts/07_figures.py
```

Runs on a MacBook Pro M2 (8 GB, Apple MPS) in a few hours, most of it super-resolution.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `scripts/common.py`.

Written outputs: `results/summary.md` (every table, generated from the data files),
`results/headline.md` (the three headline findings, every number read from a file),
`results/setB_audited.json`, `results/setA_fixed_threshold.json`, `results/psnr.csv`, and the
audit's own record — `results/audit_key.csv`, `results/audit_labels_pass1.csv`,
`results/audit_labels_pass2.csv`, `results/audit_labels.csv`.

## Data and licence

Imagery is **DOTA v1.0** (Xia et al., *DOTA: A Large-scale Dataset for Object Detection in Aerial
Images*, CVPR 2018), used via the Ultralytics-converted release, validation split only.

> **DOTA is licensed for academic and non-commercial use only.** The dataset is not redistributed
> here: `data/` is gitignored and must be downloaded from the link above. The figures in this
> repository contain small DOTA-derived crops for illustration of the method and its audit.

Detector weights are Ultralytics `yolov8s-obb.pt` (AGPL-3.0). Super-resolution weights are
Real-ESRGAN x4plus (BSD-3-Clause). The code in `scripts/` is the part of this repository offered for
reuse; point it at any super-resolver that maps an image to a 4x image and the harness runs
unchanged.
