# Does GAN super-resolution make a detector see ships that aren't there?

The one-liner: the GAN made the detector see ships that were never there about 10x more often than a plain upscale did. It found no extra real ships in return. And the standard image quality score, PSNR, didn't notice either of the issues.

I took labelled overhead imagery, degraded it 4x with a sensor model, restored it two ways — a plain
bicubic upscale and a GAN super-resolver — and ran the same pretrained ship detector on all three
versions. The detector was never retrained on super-resolved output, because that is how
super-resolution actually gets deployed on systems.

I've counted two things separately, which most super-resolution papers don't:

- **Recovery** — how many ships does it get back that the degradation destroyed?
- **Invention** — how many ships does it claim that were never there?

On 150 harbor and bridge tiles where the detector found nothing in the original image, the GAN made
it claim vessels that are not there on **14-16% of tiles**, against **1.3-2.0%** for the plain
upscale of the identical degraded data (exact McNemar p = 3.8e-6). It recovered no new ships either: ship recall
at a matched threshold was **0.881**, *below* bicubic's 0.901. PSNR saw none of it.

Every "invented" box was then checked by hand against the original image, blind, in two passes —
because my first pass at this was wrong, and I will explain how.

![Four panels: the original image, the degraded version, a bicubic upscale and the Real-ESRGAN
output. Only the last one produces ship detections — 45 of them, on a residential
hillside.](results/figures/hero.png)

---

## Why I looked at this

Super-resolution is increasingly placed in front of detection pipelines that were built and tuned on
real captures. The sharpening validation step is usually via a **pixel check**: either a
similarity score against a reference (PSNR, SSIM), or a consistency check that shrinks the output
back down to the input's resolution and compares brightness, re-running the tile if the error is too
large (radiometric audit).

**Neither of them can see objects that might have been invented.**

Example:

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
with. **A boat that is not there bypasses this brightness check.** The set of edits a downsampler cannot
see is the *null space* of that operator, and a fake vessel painted on rippling water sits right
inside it. Any check of the form "does the output still agree with the input at the input's
resolution" passes it by construction. So if we want to know whether sharpening invents objects, we're forced to go and look for the objects.

There is a second argument that makes the measurement meaningful at all. The super-resolved image is
computed from a 4x degraded copy of the original — nothing else goes in. By the **data processing
inequality**, it cannot contain more information about the scene than the original does. So a
detection that appears on the super-resolved image and *not* on the original is never new evidence.
It is either texture the model synthesized, or the same evidence re-formatted into edges and contrast
the detector happens to accept. Which of the two it is has to be decided by looking at the original —
which is exactly what the audit does.

## The setup

A satellite pixel is a measurement of how much light came off a patch of ground of size GSD (ground sample distance). What the sensor records is

$y = D\big(B(x)\big) + n$

where `x` is the true scene, `B` is the blur from the optics (the point-spread function or PSF: what a
single dot of light looks like after going through a lens), `D` averages blocks of pixels down onto
the sensor grid, and `n` is noise. Super-resolution is the inverse: given `y`, guess `x`. At 4x,
every input pixel stands for 16 output pixels, so you know one number and are guessing fifteen.

**The degradation.** I follow the sensor model from Shermeyer & Van Etten (2019):

Gaussian PSF blur (the lens)
with sigma = 0.5 x GSD_out / GSD_native (= 2.0 at 4x)

then inter-area decimation from 1024 px to 256 px. I blurred first then shrunk it, since that's the order the optics and the sensor actually happen in.

**Two restorations:** From the same 256 px image I make,

- a **bicubic x4** upscale — pure interpolation, no learned prior, no ability to invent anything;
- a **Real-ESRGAN x4plus** output — a GAN.

Bicubic is the control, and it is doing real work. Without it, "the detector found a ship on the SR
image that it missed on the original" is ambiguous: it could just mean the degradation is survivable
and any upscale would recover the object. Bicubic has access to exactly the same information and
exactly the same degraded pixels, and it cannot invent object-shaped detail, so any difference between the two paths is
attributable to the generative prior (of the GAN) and not to the resolution. 

**The detector is the control variable.** YOLOv8s-OBB pretrained on DOTAv1, oriented boxes,
`imgsz=1024`, identical weights and thresholds on all three versions, never fine-tuned on anything.
All three are scored against the same label files.

**Two test sets**, both tiled 1024x1024 from the validation split only:

- **Set A (recovery)** — 100 random tiles with at least one ship label; 3380 ship instances.
- **Set B (invention)** — 150 tiles with a harbor or bridge label, zero ship labels, and zero ship
  detections on the original at conf 0.25.

## What I found

**In one paragraph.** From the same degraded input, the GAN made the detector report a vessel
that is not there on **21–24 of 150 empty scenes (14–16%)**. The plain-interpolation control did so
on **2–3 (1.3–2.0%)**, a difference that would arise by chance about 4 times in a million. On labelled
ships, the GAN found *fewer* than the control, not more. And the standard image-quality score (PSNR)
cannot tell the tiles with phantom vessels from the clean ones.

| | Bicubic (control) | Real-ESRGAN (GAN) |
|---|---|---|
| Labelled ships found (recall, conf 0.25) | **0.901** | **0.881** |
| Empty scenes with an invented vessel (conf 0.25) | **1.3–2.0%** | **14.0–16.0%** |
| New detections that were nothing at all | 2–3 of 13 | 83–88 of 140 |
| PSNR against the original | 30.1 dB | 28.8 dB |

Every comparison uses the same detector, the same labels and the same **fixed confidence threshold**
for all three versions of each image. The detector scores each box from 0 to 1, and only boxes scoring
at least the threshold $\tau$ count.

---

### 1. Recovery: sharper did not mean more ships found

*Set A: 100 tiles, 3,380 labelled ships.*

**Matching.** A detected box $A$ is a hit on a labelled box $B$ when they overlap by at least half:

$$\text{IoU}(A,B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} \;\ge\; 0.5$$

Boxes are matched greedily, highest confidence first, and each label can be claimed only once.

**Scores.**

$$P = \frac{\text{matched boxes}}{\text{all reported boxes}} \qquad
R = \frac{\text{matched labels}}{\text{all labels}} \qquad
F_1 = \frac{2PR}{P + R}$$

- **Precision** $P$ asks: when it says "ship", is it right?
- **Recall** $R$ asks: of the real ships, how many did it find?
- **mAP50** is the area under the precision–recall curve as $\tau$ sweeps from 1 to 0, at IoU ≥ 0.5. It is a threshold-free summary: $\text{AP} = \int_0^1 P(R)\,dR$.

| $\tau$ | Version | $P$ | $R$ | $F_1$ | mAP50 |
|---|---|---|---|---|---|
| 0.25 | Original | 0.890 | 0.951 | 0.920 | 0.982 |
| 0.25 | Bicubic x4 | 0.874 | 0.901 | 0.887 | 0.946 |
| 0.25 | Real-ESRGAN x4 | 0.845 | 0.881 | 0.863 | 0.934 |
| 0.50 | Original | 0.913 | 0.935 | 0.924 | — |
| 0.50 | Bicubic x4 | 0.928 | 0.853 | 0.889 | — |
| 0.50 | Real-ESRGAN x4 | 0.916 | 0.825 | 0.868 | — |

Degradation costs the detector $0.951 - 0.901 = 0.050$ of recall. The GAN exists to win that back.
Instead it loses more:

$$\Delta R_{\text{GAN} - \text{bicubic}} = 0.881 - 0.901 = -0.020 \;\;(\tau = 0.25), \qquad 0.825 - 0.853 = -0.028 \;\;(\tau = 0.50)$$

Its output *looks* sharper and is *worse* for the detector. This matches GeoSR-Bench (2026), which
found GAN and diffusion super-resolvers sometimes underperforming no upscaling at all.

The labels omit small craft. Even the original image produces 397 ship detections with no matching
label, so an unmatched box is not proof of an error. All three versions are scored against the same
labels, so the *comparison* between them is unaffected.

---

### 2. Invention: the GAN draws vessels that are not there

*Set B: $N = 150$ harbor and bridge tiles with no ship labels and zero ship detections on the
original.* Any ship the detector reports here after degrading and restoring needs explaining.

**Defining "invented".** A detection-free original does not prove a vessel-free scene, because some
small boats are both unlabelled and missed. So every new box from **both** arms (153 in total) was
cropped from the *original* image and judged by hand: `ship`, `not_ship` or `unsure`. The judging was
blind to which arm produced the box, and done in two passes with more context the second time (see
*How I audited it*).

**The invention rate** is the fraction of tiles carrying at least one invented box:

$$r = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\big[\,\text{tile } i \text{ has} \ge 1 \text{ invented box}\,\big]$$

Here $\mathbf{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. Each tile counts at most once, so
no single scene can dominate. The rate is reported as a band:

- $r_{\text{low}}$ counts only `not_ship` boxes as invented.
- $r_{\text{high}}$ counts `not_ship` and `unsure` boxes as invented.

**Confidence intervals** come from a bootstrap. Draw 150 tiles *with replacement*, recompute $r$,
and repeat $B = 10{,}000$ times:

$$\text{95\% CI} = \big[\, r^{*}_{(2.5\%)},\; r^{*}_{(97.5\%)} \,\big]$$

**Significance** uses the exact McNemar test, which is built for paired data: every tile is restored both
ways. Tiles where both arms agree carry no information about which is worse, so only the discordant ones
count:

- $b$ = tiles where only the GAN invented
- $c$ = tiles where only bicubic invented
- $n = b + c$

If both arms were equally prone to invent, each discordant tile would be a fair coin flip, so

$$p = 2\sum_{k=0}^{\min(b,c)} \binom{n}{k}\left(\tfrac{1}{2}\right)^{n}$$

At the low end, $b = 19$ and $c = 0$: every tile where bicubic invented, the GAN did too. Then

$$p = 2\left(\tfrac{1}{2}\right)^{19} = \frac{2}{524{,}288} \approx 3.8 \times 10^{-6}$$

That is the probability of 19 heads in a row from a fair coin, counting either side.

**The headline result:**

| $\tau$ | band | Real-ESRGAN | Bicubic | $r_{\text{GAN}} - r_{\text{bic}}$ | $p$ |
|---|---|---|---|---|---|
| 0.25 | low | **14.0%** [8.7, 20.0] | 1.3% [0.0, 3.3] | +12.7 pts | **3.8 × 10⁻⁶** |
| 0.25 | high | **16.0%** [10.0, 22.0] | 2.0% [0.0, 4.7] | +14.0 pts | **5.7 × 10⁻⁶** |
| 0.50 | low | 2.7% [0.7, 5.3] | 0.7% [0.0, 2.0] | +2.0 pts | 0.375 |
| 0.50 | high | 3.3% [0.7, 6.7] | 0.7% [0.0, 2.0] | +2.7 pts | 0.219 |

**At $\tau = 0.50$ the difference is not significant.** Only 31 boxes survive the stricter threshold,
too few to resolve a 2-point gap. The raw firing rate at that threshold still differs significantly:
9.3% vs 1.3%, $p = 0.004$. That rate counts every new ship box with no human judgement at all. A
stricter threshold reduces invention. It does not remove it.

**One scene, and why it doesn't drive the result.** A single dark residential hillside produced 45 of
the 83 non-vessel boxes. The GAN turned soft rooftops into crisp bright rectangles in rows, which is
what ships look like to this detector. Because $r$ counts *tiles*, that scene contributes exactly one
event. Removing it from the numerator and the denominator:

$$r_{\text{low}} = \frac{21}{150} = 14.0\% \;\;\longrightarrow\;\; \frac{20}{149} = 13.4\% \quad (p = 7.6 \times 10^{-6})$$

**What the new detections actually were** ($\tau = 0.25$):

| | New boxes | Real vessel | Nothing there | Unsure |
|---|---|---|---|---|
| Real-ESRGAN | 140 | 52 | 83 | 5 |
| Real-ESRGAN, hillside removed | 95 | 52 | 38 | 5 |
| Bicubic | 13 | 10 | 2 | 1 |

The GAN is not useless. It surfaced 52 real boats that the labels missed and the detector missed on
the original. The issue is the price, measured as phantoms per real find:

$$\phi = \frac{\text{nothing there}}{\text{real vessel}}, \qquad
\phi_{\text{GAN}} = \frac{83}{52} = 1.60 \;\;\text{or}\;\; \frac{38}{52} = 0.73 \text{ without the hillside}, \qquad
\phi_{\text{bic}} = \frac{2}{10} = 0.20$$

On the control, a newly confident detection is usually a real boat. On the GAN, it is close to a coin
flip. *(Boxes cluster within scenes, so these counts describe; the tile-level table above is what tests.)*

---

### 3. PSNR cannot see it

PSNR scores an image $\hat{x}$ against the original $x$ by mean squared pixel error. Higher means closer:

$$\text{MSE} = \frac{1}{HW}\sum_{u=1}^{H}\sum_{v=1}^{W}\big(x_{uv} - \hat{x}_{uv}\big)^2, \qquad
\text{PSNR} = 10\log_{10}\frac{255^2}{\text{MSE}}$$

SSIM compares local means $\mu$, variances $\sigma^2$ and covariance $\sigma_{xy}$:

$$\text{SSIM}(x,\hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$

| Version | PSNR | SSIM |
|---|---|---|
| Bicubic x4 | 30.1 dB | 0.753 |
| Real-ESRGAN x4 | 28.8 dB | 0.748 |

A 1.3 dB gap sits beside a roughly tenfold gap in invented vessels.

**Why the score can't see a phantom.** Here is an illustrative case. A tile with PSNR 28.8 dB has
$\text{MSE} = 255^2 / 10^{2.88} \approx 86$. Paint a 50-pixel phantom, each pixel off by 80 levels,
into a $1024 \times 1024$ tile:

$$\Delta\text{MSE} = \frac{50 \times 80^2}{1024^2} \approx 0.31, \qquad
\Delta\text{PSNR} = 10\log_{10}\frac{86}{86.31} \approx -0.016 \text{ dB}$$

A vessel that does not exist moves the score by about one-sixtieth of a decibel.

**Within the GAN's own output, PSNR fails to identify which tiles carry a phantom.**

- **AUC = 0.59** ($p = 0.17$), where $\text{AUC} = \Pr\big(\text{PSNR}_{\text{phantom tile}} < \text{PSNR}_{\text{clean tile}}\big)$ for a random pair. A value of 0.5 is a coin toss.
- **Spearman $\rho_s = -0.11$** ($p = 0.17$), where $\rho_s = 1 - \dfrac{6\sum_i d_i^2}{n(n^2-1)}$ and $d_i$ is the gap between tile $i$'s PSNR rank and its invention rank. A value of 0 means no relationship.

**A result I nearly published, and why it was wrong.** Before the hand audit, PSNR looked predictive:
AUC 0.68, $p = 0.0005$. The cause was a confound, a third variable driving both measurements:

$$\text{busy scene} \;\Rightarrow\; \text{low PSNR} \quad(\rho_s = -0.44,\; p = 2 \times 10^{-8}), \qquad
\text{busy scene} \;\Rightarrow\; \text{more unlabelled real boats}$$

So "low PSNR predicts invention" meant "low PSNR predicts clutter". Once each box was checked against
the original, the relationship disappeared. Counting objects without looking at them reproduces that
mistake, which is why the audit exists.

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
