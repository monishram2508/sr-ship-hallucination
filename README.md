# Does GAN super-resolution make a detector see ships that aren't there?

I took labelled overhead imagery, degraded it 4x with a sensor model, restored it two ways — a plain
bicubic upscale and a GAN super-resolver — and ran the **same** pretrained ship detector on all three
versions. The detector was never retrained on super-resolved output, because that is how
super-resolution actually gets deployed: bolted in front of a pipeline that already exists.

Then I counted two things separately, which most super-resolution papers do not:

- **Recovery** — how many ships does it get back that the degradation destroyed?
- **Invention** — how many ships does it claim that were never there?

On 150 harbor and bridge tiles where the detector found nothing in the original image, the GAN made
it claim vessels that are not there on **14-16% of tiles**, against **1.3-2.0%** for the plain
upscale of the identical degraded data (exact McNemar p = 3.8e-06). It recovered nothing: ship recall
at a matched threshold was **0.881**, *below* bicubic's 0.901. PSNR saw none of it.

Every "invented" box was then checked by hand against the original image, blind, in two passes —
because my first pass at this was wrong, and I will explain how.

![Four panels: the original image, the degraded version, a bicubic upscale and the Real-ESRGAN
output. Only the last one produces ship detections — 45 of them, on a residential
hillside.](results/figures/hero.png)

---

## Why I looked at this

Super-resolution is increasingly placed in front of detection pipelines that were built and tuned on
real captures. The way the sharpening step usually gets validated is a **pixel check**: either a
similarity score against a reference (PSNR, SSIM), or a consistency check that shrinks the output
back down to the input's resolution and compares brightness, re-running the tile if the error is too
large.

**Neither kind of check can see objects.** Here is the arithmetic, which is the whole reason this
project exists.

Take one pixel of a 1 m input image: a patch of open water with brightness 120. At 4x, the model has
to return 16 numbers where there was one. The honest answer is flat:

```
120 120 120 120
120 120 120 120
120 120 120 120
120 120 120 120
```

But so is this one — a bright hull with its shadow underneath:

```
120 200 200 120
120 200 200 120
120  40  40 120
120  40  40 120
```

Both average to exactly 120. Shrink either one back down and you get the input pixel you started
with. **A boat that is not there costs nothing in brightness.** The set of edits a downsampler cannot
see is the *null space* of that operator, and a fake vessel painted on rippling water sits right
inside it. Any check of the form "does the output still agree with the input at the input's
resolution" passes it by construction.

So if you want to know whether sharpening invents objects, you have to go and look for the objects.
That is what this repository does.

There is a second argument that makes the measurement meaningful at all. The super-resolved image is
computed from a 4x degraded copy of the original — nothing else goes in. By the **data processing
inequality**, it cannot contain more information about the scene than the original does. So a
detection that appears on the super-resolved image and *not* on the original is never new evidence.
It is either texture the model synthesized, or the same evidence re-formatted into edges and contrast
the detector happens to accept. Which of the two it is has to be decided by looking at the original —
which is exactly what the audit does.

## The setup

**The forward problem.** A satellite pixel is a measurement of how much light came off a patch of
ground, and the size of that patch is the ground sample distance. What the sensor records is

```
y = D( B(x) ) + n
```

where `x` is the true scene, `B` is the blur from the optics (the point-spread function — what a
single dot of light looks like after going through a lens), `D` averages blocks of pixels down onto
the sensor grid, and `n` is noise. Super-resolution is the inverse: given `y`, guess `x`. At 4x,
every input pixel stands for 16 output pixels, so you know one number and are guessing fifteen.
Different models guess differently, and **how** they guess is the whole story.

**The degradation.** I follow the sensor model from Shermeyer & Van Etten (2019): Gaussian PSF blur
with sigma = 0.5 x GSD_out / GSD_native (= 2.0 at 4x), then inter-area decimation from 1024 px to
256 px. Blur first, then shrink — that order matters, because it is the order the optics and the
sensor actually happen in. Everything is written as PNG; JPEG artefacts would confound the whole
comparison.

**Two restorations, and why two.** From the same 256 px image I make:

- a **bicubic x4** upscale — pure interpolation, no learned prior, no ability to invent anything;
- a **Real-ESRGAN x4plus** output — a GAN.

Bicubic is the control, and it is doing real work. Without it, "the detector found a ship on the SR
image that it missed on the original" is ambiguous: it could just mean the degradation is survivable
and any upscale would recover the object. Bicubic has access to exactly the same information and
exactly the same degraded pixels, and it cannot hallucinate — so any gap between the two arms is
attributable to the generative prior and not to the resolution.

**The detector is the control variable.** YOLOv8s-OBB pretrained on DOTAv1, oriented boxes,
`imgsz=1024`, identical weights and thresholds on all three versions, never fine-tuned on anything.
All three are scored against the same label files. If the comparison moves, the super-resolution is
what moved it.

**Two test sets**, both tiled 1024x1024 from the validation split only:

- **Set A (recovery)** — 100 random tiles with at least one ship label; 3380 ship instances.
- **Set B (invention)** — 150 tiles with a harbor or bridge label, zero ship labels, and zero ship
  detections on the original at conf 0.25.

## What I found

### 1. It recovered nothing

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

The GAN sits **2.0 points of recall below a plain bicubic upscale** at conf 0.25, and 2.8 points
below at conf 0.50. Both are well below the original. So the thing it is sold for — getting back what
resolution destroyed — did not happen here, and the sharper-looking image was actively worse for the
detector than the blurry one. That is consistent with GeoSR-Bench (UMD, 2026), which found GAN and
diffusion super-resolvers sometimes doing worse downstream than not upscaling at all.

A note on the word I do **not** use: "false positive". The original imagery itself leaves 397 ship
detections unmatched against these labels, because DOTA omits small craft. A detection with no
matching label is not proof the detector was wrong — so the column is called *unmatched detections*,
which is what it measures. All three versions are scored against the same labels, so the comparison
between them survives the labels being incomplete.

### 2. It invented ships

On 150 tiles where the detector produced **zero** ship detections on the original image.

Every test here is **tile-level**: a tile contributes at most one event, so no single pathological
scene can run away with the rate. Confidence intervals are percentile bootstrap over 10,000 resamples
of the 150 tiles (seed 0); p-values are the exact binomial McNemar test on the paired tiles, exact
rather than chi-square because one discordant cell is zero.

**(a) The firing rate — no human judgement in it at all.** How often did degrading and restoring make
the detector fire where the sharp original was silent?

| conf | Version | tiles | % of tiles | 95% CI | paired difference | exact McNemar p |
|---|---|---|---|---|---|---|
| 0.25 | Real-ESRGAN x4 | 43/150 | **28.7%** | [21.3, 36.0] | +22.7 pp [16.0, 29.3] | **1.2e-10** |
| 0.25 | Bicubic x4 | 9/150 | 6.0% | [2.7, 10.0] | — | — |
| 0.50 | Real-ESRGAN x4 | 14/150 | **9.3%** | [4.7, 14.0] | +8.0 pp [3.3, 13.3] | **4.2e-03** |
| 0.50 | Bicubic x4 | 2/150 | 1.3% | [0.0, 3.3] | — | — |

**(b) Invention — tiles carrying at least one box that a blind audit judged *not a vessel*.** The
lower band counts only boxes judged not-a-vessel; the upper band also counts the handful still
uncertain after two passes. It is a band, not a confidence interval: both ends are reported because
picking one would be picking a result.

| conf | band | Real-ESRGAN | Bicubic | difference | exact McNemar p |
|---|---|---|---|---|---|
| 0.25 | lower | **14.0%** [8.7, 20.0] | 1.3% [0.0, 3.3] | +12.7 pp | **3.8e-06** |
| 0.25 | upper | **16.0%** [10.0, 22.0] | 2.0% [0.0, 4.7] | +14.0 pp | **5.7e-06** |
| 0.50 | lower | 2.7% [0.7, 5.3] | 0.7% [0.0, 2.0] | +2.0 pp | 0.375 |
| 0.50 | upper | 3.3% [0.7, 6.7] | 0.7% [0.0, 2.0] | +2.7 pp | 0.219 |

**At conf 0.50 this is not significant, and I am not going to dress that up.** Only 31
super-resolved boxes survive the higher threshold at all, 12-13 of them not-a-vessel on 4-5 tiles —
too little data to resolve a 2-point difference. The point estimate still moves the same way, and the
judgement-free firing rate *is* significant at that threshold (p = 4.2e-03). Raising the detector's
confidence bar reduces invention. It does not remove it.

**One tile dominates, and the result survives removing it.** A single dark residential hillside
carries 45 of the 85 not-a-vessel boxes in the whole set. Dropping that tile from the numerator *and*
the denominator leaves 149 tiles:

| conf 0.25 band | all 150 tiles | worst tile dropped (149) | bicubic | exact McNemar p |
|---|---|---|---|---|
| lower | 14.0% | **13.4%** [8.1, 19.5] | 1.3% | **7.6e-06** |
| upper | 16.0% | **15.4%** [10.1, 21.5] | 2.0% | **1.1e-05** |

It moves by 0.6 points, because a tile-level test already counts each tile once. That is the reason
the headline is tile-level and not box-level.

**(c) The two arms fail in different ways.** This is the figure that *does* depend on that one tile,
so both versions are always given together.

| extra boxes where the original was silent (conf 0.25) | total | real vessel | no vessel | uncertain |
|---|---|---|---|---|
| Real-ESRGAN x4 | 140 | 52 | 83 | 5 |
| Real-ESRGAN x4, worst tile dropped | 95 | 52 | 38 | 5 |
| Bicubic x4 | 13 | 10 | 2 | 1 |

59-63% of the GAN's extra boxes had nothing under them set-wide, 40-45% with the worst tile dropped.
Bicubic's 13 extra boxes contained 2-3 with nothing under them — 13 boxes is too few for a percentage
to carry meaning, so it is given as a count. The other 10 were **real unlabelled vessels**: boats
DOTA never annotated that the detector missed on the original and found after upscaling. That is a
genuine recovery.

So: when a plain upscale makes the detector newly confident, it is usually right. When the GAN does
it, most of the time there is nothing there. Same detector, same tiles, same threshold, opposite
reliability — and that contrast is the actual result, more than any single percentage.

### 3. PSNR saw none of it

| Version | mean PSNR vs original | mean SSIM |
|---|---|---|
| Bicubic x4 | 30.12 dB | 0.753 |
| Real-ESRGAN x4 | 28.78 dB | 0.748 |

A 1.3 dB gap, while the tile-level invention rates differ by roughly 10x. The pixel scores happen to
rank bicubic first, but nothing in the number distinguishes "slightly less faithful texture" from
"draws vessels that do not exist". Within the super-resolved arm, PSNR cannot even rank the tiles:

![Per-tile PSNR against audited invented objects: no relationship, AUC
0.59](results/figures/psnr_vs_detection_delta.png)

Low PSNR predicts a tile carrying an invented object with AUC 0.59 (p = 0.17), Spearman rho = -0.11
(p = 0.17). A coin flip.

**And this is where I got it wrong the first time.** Before the audit existed, the same plot looked
genuinely predictive — AUC 0.68, p = 0.0005 — and I nearly wrote that up. It was a confound. PSNR
tracks how *busy* a scene is (Spearman rho = -0.44 against labelled object count, p = 2e-08), and
busy scenes are exactly where the unlabelled boats live. So "low PSNR predicts invention" was really
"low PSNR predicts clutter, and clutter is where the real unlabelled vessels are". Counting invented
objects without going and looking at them reproduces that mistake, and most of the first version of
this project *was* that mistake. See the audit section.

## Why the GAN does this

A GAN is two networks trained against each other. The **generator** takes the blurry image and paints
a sharp version. The **discriminator** looks at an image and guesses whether it was generated or is a
real sharp photo — it checks for crisp edges, strong contrast, plausible texture. It does not know
what a ship is, and it is not asked whether the content is true.

That is the whole problem: **the generator is rewarded for looking real, not for being right.** If
painting something that reads as a convincing object makes the output harder to tell from a real
photograph, that is reward.

The hero figure above is that mechanism caught in the act. The input is a dark greyscale hillside
where houses are soft bright blobs. Real-ESRGAN sharpens those blobs into crisp, high-contrast bright
rectangles with dark edges, lined up along the streets. The detector was trained on DOTA, where ships
seen from above are exactly that: bright elongated rectangles with dark water around them, in rows.
So it fires — 45 times, on one tile, with confidences up to 0.72. On the original and on the bicubic
upscale the houses stay soft enough that the pattern never matches, and both return zero.

Nobody involved made an error. The super-resolver did its job, the detector did its job, and the
output is a confident list of 45 vessels on a hillside.

## How I audited it, and why the audit exists

The first version of Set B was built on an assumption I wrote down and that turned out to be false:
that a tile with zero ship labels *and* zero ship detections on the original contains no ships. It
verifies zero **detections**, not zero **vessels**. DOTA omits small craft, so some of those tiles do
contain boats nobody labelled and the detector missed. Before auditing, the headline said
super-resolution invented on 28.7% of tiles. About a third of those were real boats. That number was
wrong, and the only way to find out was to look at all 153 crops by hand.

So the protocol is built to make my own verdicts hard to bend:

1. **Every extra box gets audited, both arms.** All 153 boxes either version produced on Set B at
   conf 0.25 go into one pool. Auditing only the GAN's boxes would manufacture the asymmetry the
   project is trying to measure.
2. **Crops come from the original image**, never from the version that produced the box. The question
   is whether a vessel is there, and the original is the best evidence of that.
3. **Blind.** The pool is split into `audit_key.csv` (crop id -> version, tile, confidence, box) and
   `audit_labels.csv` (crop id -> verdict). Order is shuffled with a fixed seed *before* ids are
   assigned, so an id leaks nothing — not the version, not the tile, not the confidence. The
   labelling tool never opens the key.
4. **Three verdicts, and a blank is a hard error.** `ship` / `not_ship` / `unsure`. The recount script
   exits if any crop is unlabelled instead of defaulting it, because a silent default manufactures a
   result.
5. **Two bands instead of a judgement call.** Lower bound counts only `not_ship`, upper also counts
   `unsure`, and every invention number is reported as that range.
6. **Two passes.** Pass 1 showed 48 px of context around each box. The 38 crops I marked `unsure` were
   re-rendered with 128 px, reshuffled under a new seed and judged again, still blind. They resolved
   **25 to not_ship, 7 to ship, 6 still uncertain** — which narrowed the invention band from 6.0-18.0%
   to 14.0-16.0%. Pass 1 is kept at `results/audit_labels_pass1.csv` and both passes are reported
   side by side.
7. **The recount is checked against the raw counts.** Forcing every verdict to `not_ship` reproduces
   the pre-audit detection counts exactly, per tile, zero mismatches — so the join between key, labels
   and counts is sound.

Contact sheets for every verdict are in `results/figures/audit/`, so the labelling can be checked
rather than trusted:

![Contact sheet of every crop judged not a vessel, with a magenta box on each claimed
detection](results/figures/audit/qa_not_ship.png)

## What this does and does not say

1. **The degradation is simulated.** Gaussian PSF blur plus decimation is not a real coarser sensor:
   no atmosphere, no real optics, no sensor noise model, no compression.
2. **One super-resolver, one detector.** Real-ESRGAN x4plus and YOLOv8s-OBB. This is evidence about
   GAN super-resolution as a class, not proof about any particular model — and Real-ESRGAN is a GAN
   trained on everyday photographs, not a satellite-trained model.
3. **A GAN is the worst case on purpose, and a diffusion model with guardrails would not behave
   identically.** A production diffusion super-resolver typically adds a data-consistency step at
   every denoising iteration, which nudges the guess so that shrinking it back down keeps matching the
   input. That constrains geometry — buildings cannot move, coastlines cannot shift — and it should
   reduce this failure. It does not eliminate it, because of the arithmetic at the top of this file:
   a data-consistency step enforces `D(output) = input`, and the fake-boat edit satisfies
   `D(output + n) = D(output)` exactly. It is inside the null space of the very operator doing the
   constraining. **Which also means a separate brightness audit after the fact is close to redundant
   with the consistency step** — both enforce agreement at the input's resolution, so a model that
   already passes one passes the other almost by construction. The number this repository reports is
   an upper bound for a guarded model; the right experiment is to run this harness on the guarded
   model and get its number.
4. **The detector may have seen these images in training**, since it is pretrained on the dataset
   whose validation split I use. The original-image scores are an upper bound. The relative comparison
   between the three versions is unaffected — same detector, same labels, same thresholds.
5. **Single labeller: me.** Blind to version, tile and confidence, but obviously not blind to what the
   project is testing. Two mitigations, neither a substitute for a second labeller: both arms were
   audited in the same blind pool, and every number is a band rather than a point. The direction of
   the second pass is worth stating plainly — with more context the uncertain crops resolved 25 to
   not-a-vessel against 7 to vessel, which moved the lower band *up*.
6. **Invention at conf 0.50 is underpowered.** 31 surviving boxes cannot resolve a 2-point difference.
   Reported as non-significant, not as null.
7. **Boxes cluster within tiles**, so boxes are not independent observations. Every significance test
   is therefore tile-level; box-level figures are descriptive composition only, and always shown
   alongside the version with the dominant tile removed.
8. **The dataset's labels are incomplete** — it omits small craft — which is the entire reason the
   audit exists, and the reason "unmatched detection" replaces "false positive" throughout.
9. **The dataset mixes aerial and satellite imagery** at varying ground sample distances, so "4x
   degradation" is not one physical scale.

## Related work

- **Shermeyer & Van Etten (2019), [arXiv:1812.04098](https://arxiv.org/abs/1812.04098).**
  Super-resolution for object detection on 30 cm satellite imagery, with VDSR and a random-forest
  super-resolver. Their detectors were **retrained on super-resolved output**, which is the opposite
  of the setting here — this project asks what happens when SR is bolted in front of a detector nobody
  retrained. Their sensor-degradation model is the one I use.
- **GeoSR-Bench, Li et al. (2026), [arXiv:2605.00310](https://arxiv.org/abs/2605.00310).** Nine
  super-resolution models across transformer, neural-operator, GAN and diffusion families on
  downstream geospatial tasks. PSNR and SSIM often fail to track downstream performance and sometimes
  correlate negatively; in places ESRGAN and a diffusion model did worse than no upscaling at all.
  Their downstream tasks are **pixel-level** — segmentation and regression — so there is no object
  detection and no count of invented objects.

**What is new here:** a direct count of invented *objects* in scenes verified empty of detections,
with a detector that was not retrained, and with every claimed object adjudicated by a blind hand
audit instead of assumed from a label file.

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
.venv/bin/python scripts/08_audit_boxes.py    # crops + key + blank label file
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

Runs on a MacBook Pro M2 (8 GB, Apple MPS) in a few hours, most of it super-resolution.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `scripts/common.py`.

**Every number in the write-up is generated from a file in `results/`; none is typed by hand.** The
two write-ups (`results/summary.md`, `results/headline.md`) are produced by scripts that read the
JSON and CSV outputs, so the prose cannot drift away from the data. The audit's own record —
`audit_key.csv`, `audit_labels_pass1.csv`, `audit_labels_pass2.csv`, `audit_labels.csv` — is in the
repository too, so anyone can disagree with my verdicts and recount.

## Data and licence

Imagery is **DOTA v1.0** (Xia et al., *DOTA: A Large-scale Dataset for Object Detection in Aerial
Images*, CVPR 2018), used via the Ultralytics-converted release, validation split only.

> **DOTA is licensed for academic and non-commercial use only.** The dataset is not redistributed
> here: `data/` is gitignored and must be downloaded from the link above. The figures in this
> repository contain small DOTA-derived crops to illustrate the method and its audit.

Detector weights are Ultralytics `yolov8s-obb.pt` (AGPL-3.0). Super-resolution weights are
Real-ESRGAN x4plus (BSD-3-Clause). The code in `scripts/` is the part offered for reuse: point it at
anything that maps an image to a 4x image and the harness runs unchanged.
