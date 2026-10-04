# Results

Set A = 99 tiles carrying 3378 ship instances. Set B = 150 harbor/bridge tiles with zero ship labels AND zero ship detections on the original at conf 0.25 — **tiles where the detector saw nothing in the original**.

Set B was built as "verified empty" and that term is not used here: the audit below shows the construction verifies zero HR *detections*, not zero vessels. DOTA omits small craft, so some of these tiles do contain boats that were never labelled and that the detector missed on the original. Every claim below is therefore made against a hand audit of the original imagery, not against the label file.

Set A was sampled as 100 tiles; ultralytics drops 1 (P0261__1024__390___0.png) because a DOTA polygon vertex falls outside the crop (coordinate 1.179). The same tile is dropped from all three versions, so the comparison is unaffected.

The Set B columns in this first table are **raw detection counts**, kept because the project brief asked for them; they are superseded by the audited numbers further down.

| Version | Set A ship P | Set A ship R | Set A mAP50 | Set B ship detections / 100 tiles (conf 0.25) | (conf 0.5) | Mean PSNR vs HR |
|---|---|---|---|---|---|---|
| HR | 0.960 | 0.940 | 0.982 | 0 (by construction) | 0.0 | — |
| Bicubic | 0.941 | 0.907 | 0.946 | 8.7 | 1.3 | 30.12 dB |
| Real-ESRGAN | 0.953 | 0.820 | 0.934 | 93.3 | 20.7 | 28.78 dB |

## Set B BEFORE the audit — raw detection counts (superseded)

These are every ship detection, with no check on whether an object was really there. The audit below shows a third of them were real unlabelled vessels, so **do not quote this table as an invention rate** — it is kept because it is what the detector did.

Tile-level, not ship-level: one tile contributes at most one event, so a single pathological tile cannot drive the number. CIs are percentile bootstrap over 10,000 resamples of the 150 tiles (seed 0); the tile is the unit of independence.

| Version | conf | tiles with >=1 invented ship | **% of tiles** | 95% CI | total invented ships | per 100 tiles |
|---|---|---|---|---|---|---|
| HR | 0.25 | 0 | **0.0%** | [0.0, 0.0] | 0 | 0.0 |
| HR | 0.5 | 0 | **0.0%** | [0.0, 0.0] | 0 | 0.0 |
| Bicubic | 0.25 | 9 | **6.0%** | [2.7, 10.0] | 13 | 8.7 |
| Bicubic | 0.5 | 2 | **1.3%** | [0.0, 3.3] | 2 | 1.3 |
| Real-ESRGAN | 0.25 | 43 | **28.7%** | [21.3, 36.0] | 140 | 93.3 |
| Real-ESRGAN | 0.5 | 14 | **9.3%** | [4.7, 14.0] | 31 | 20.7 |

### SR vs bicubic, paired on the same tiles

| conf | SR-only tiles (b) | bicubic-only tiles (c) | difference in % of tiles | paired 95% CI | exact McNemar p |
|---|---|---|---|---|---|
| 0.25 | 34 | 0 | +22.7 pp | [16.0, 29.3] | **1.16e-10** |
| 0.5 | 14 | 2 | +8.0 pp | [3.3, 13.3] | **0.0042** |

McNemar is exact (binomial), not chi-square: one discordant cell is 0 at both thresholds, where the chi-square approximation is not valid.


## Pixel-similarity scores (all tiles, both sets)

| Version | n tiles | mean PSNR | mean SSIM |
|---|---|---|---|
| Bicubic | 250 | 30.12 dB | 0.7533 |
| Real-ESRGAN | 250 | 28.78 dB | 0.7477 |

## Set B AFTER the blind audit — the two metrics that count

Every ship box on Set B at conf 0.25 (153 crops: SR and bicubic both) was cropped from the **HR** tile (pad 48 px, nearest-neighbour enlargement) and judged by hand as `ship` / `not_ship` / `unsure`. The audit is **blind**: crop order was shuffled with a fixed seed and crop ids carry no version, tile or confidence (`results/audit_key.csv` holds the mapping, `results/audit_labels.csv` the verdicts, contact sheets in `results/figures/audit/`). Boxes judged `ship` are unlabelled vessels DOTA omitted and the HR detector missed — recoveries, not inventions — and do not count. `unsure` counts only in the upper band.

Verdict tally: `bicubic_not_ship` 2 · `bicubic_ship` 10 · `bicubic_unsure` 1 · `sr_not_ship` 83 · `sr_ship` 52 · `sr_unsure` 5

### (a) INVENTION — % of tiles with >=1 box judged *not a vessel*

| band | SR % of tiles | SR 95% CI | bicubic % of tiles | difference | paired 95% CI | exact McNemar p |
|---|---|---|---|---|---|---|
| conf0.25_lower | 14.0% | [8.7, 20.0] | 1.3% | +12.7 pp | [7.3, 18.0] | 3.81e-06 |
| conf0.25_upper | 16.0% | [10.0, 22.0] | 2.0% | +14.0 pp | [8.0, 20.0] | 5.72e-06 |
| conf0.5_lower | 2.7% | [0.7, 5.3] | 0.7% | +2.0 pp | [-0.7, 5.3] | 0.3750 |
| conf0.5_upper | 3.3% | [0.7, 6.7] | 0.7% | +2.7 pp | [0.0, 6.0] | 0.2188 |

**This is the lead result.** SR exceeds bicubic at conf 0.25 in both bands — 14.0% of tiles against 1.3% (p = 3.81e-06) counting only boxes judged not-a-vessel, and 16.0% against 2.0% (p = 5.72e-06) counting the uncertain ones too.

**It is not significant at conf 0.50** (p = 0.375 lower, 0.219 upper), where only 31 SR boxes survive the threshold at all. That is an absence of power, not evidence of absence: the point estimate still moves the same way (3.3% vs 0.7%). Raising the detector threshold reduces invention; it does not remove it.

Invention is **concentrated**: of the 85 boxes judged not-a-vessel across the whole set, 45 sit on a single tile (`P1751__1024__0___2976`), where SR turned a dark residential hillside into 45 "ships" and both the original and the bicubic upscale returned zero. The sensitivity analysis below removes that tile entirely.


#### Pass 1 vs pass 2 — adjudicating the `unsure` crops

Every crop labelled `unsure` in pass 1 was re-rendered with 128 px of context instead of 48, reshuffled under a new seed, and judged again, still blind. A pass-2 verdict replaces the pass-1 `unsure` and nothing else changes. Pass 1 is kept at `results/audit_labels_pass1.csv`.

| verdict | pass 1 | final | change |
|---|---|---|---|
| `ship` | 55 | 62 | +7 |
| `not_ship` | 60 | 85 | +25 |
| `unsure` | 38 | 6 | -32 |

| band | pass-1 SR % | final SR % | pass-1 McNemar p | final McNemar p |
|---|---|---|---|---|
| conf0.25_lower | 6.0% | **14.0%** | 0.0391 | **3.81e-06** |
| conf0.25_upper | 18.0% | **16.0%** | 1.55e-06 | **5.72e-06** |
| conf0.5_lower | 0.7% | **2.7%** | 1.0000 | **0.3750** |
| conf0.5_upper | 4.0% | **3.3%** | 0.2891 | **0.2188** |

The second pass pulled the two bands together — the invention estimate at conf 0.25 was 6.0%-18.0% of tiles after pass 1 and is **14.0%-16.0%** after pass 2 — and strengthened the conf-0.25 result by about four orders of magnitude in p. Only 6 of 153 crops remain uncertain, so the band is now narrow enough that the choice of how to treat `unsure` barely matters.

The 38 pass-1 `unsure` crops resolved 25 to `not_ship` and 7 to `ship`, with 6 still uncertain at 128 px of context.


#### Sensitivity — the same test with the dominant tile removed

`P1751__1024__0___2976` carries 45 of the 85 not-a-vessel boxes. Dropping it **from the denominator as well as the numerator** leaves 149 tiles and 108 crops. The tile-level result is barely affected, because the test already counts each tile once:

| band | SR % (150 tiles) | SR % (149, tile dropped) | bicubic % | difference | exact McNemar p |
|---|---|---|---|---|---|
| conf0.25_lower | 14.0% | **13.4%** [8.1, 19.5] | 1.3% | +12.1 pp | **7.63e-06** |
| conf0.25_upper | 16.0% | **15.4%** [10.1, 21.5] | 2.0% | +13.4 pp | **1.10e-05** |

The **box-level** share is the figure that depends on the tile, and it should always be quoted next to the excluded version. Of SR's 140 extra boxes set-wide, 59%-63% had no vessel under them; of the 95 that remain once `P1751__1024__0___2976` is dropped, 38-43 do (40%-45%). Bicubic's comparable figure is 2-3 of 13 boxes — too few boxes for a percentage to mean anything, which is why it is given as a count.

### (b) CONFIDENCE BEYOND THE ORIGINAL — detector fires on the version, silent on HR

The supporting metric, and the only one that depends on no verdict at all: it counts tiles where degrading and restoring made the detector fire where the sharp original did not. The verdict split then says how often that extra confidence was justified.

| conf | Version | tiles | % of tiles | 95% CI | boxes | ship | not_ship | unsure |
|---|---|---|---|---|---|---|---|---|
| 0.25 | Real-ESRGAN | 43 | **28.7%** | [21.3, 36.0] | 140 | 52 (37%) | 83 (59%) | 5 (4%) |
| 0.25 | Bicubic | 9 | **6.0%** | [2.7, 10.0] | 13 | 10 (77%) | 2 (15%) | 1 (8%) |
| 0.5 | Real-ESRGAN | 14 | **9.3%** | [4.7, 14.0] | 31 | 18 (58%) | 12 (39%) | 1 (3%) |
| 0.5 | Bicubic | 2 | **1.3%** | [0.0, 3.3] | 2 | 1 (50%) | 1 (50%) | 0 (0%) |

Paired at conf 0.25: SR-only tiles 34, bicubic-only 0, difference +22.7 pp [16.0, 29.3], exact McNemar p = 1.16e-10.

**Why this is the right frame.** The SR image is computed from a 4x degraded copy of the original, so by the data processing inequality it cannot contain more information about the scene than the original does. A detection present on SR and absent on HR is never new evidence — it is synthesized texture, or the same evidence re-formatted into edges the detector's priors accept. The audit says which: about a third of the time there really was a vessel the original detector missed; the rest of the time there was not.


## Footnote — what the Set A precision and recall are measured AT

Set A P and R are **not** at a fixed confidence threshold. Ultralytics' `ap_per_class` (ultralytics/utils/metrics.py:92) chooses one index `i = smooth(f1_curve.mean(0), 0.1).argmax()` — the confidence that maximises the **smoothed mean F1 across all 15 classes** — and then reads *every* class's P and R at that same index. So the ship row is quoted at whatever confidence happened to be best for the 15-class average, and that confidence is **different for each version**:

| Version | confidence at which P and R are read |
|---|---|
| HR | 0.627 |
| Bicubic | 0.338 |
| Real-ESRGAN | 0.565 |

**Consequence: the P/R columns above are not directly comparable across versions.** Bicubic's recall is read at conf 0.338 and Real-ESRGAN's at conf 0.565 — a lower bar for bicubic. `mAP50` is threshold-free and *is* comparable; so is the fixed-threshold table in `results/setA_fixed_threshold.json`, which re-measures P/R/F1 for all three versions at the same confidence.


## Set A at a FIXED confidence — the comparable version of the recovery table

Ship class only. Detections matched to ground truth greedily, highest confidence first, rotated-box IoU >= 0.5 (ultralytics `batch_probiou`). Same threshold for every version, so these rows *can* be compared.

| conf | Version | TP | Unmatched detections | Missed GT | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.25 | HR | 3216 | 397 | 164 | 0.890 | 0.951 | 0.920 |
| 0.25 | Bicubic | 3046 | 439 | 334 | 0.874 | 0.901 | 0.887 |
| 0.25 | Real-ESRGAN | 2977 | 546 | 403 | 0.845 | 0.881 | 0.863 |
| 0.5 | HR | 3159 | 302 | 221 | 0.913 | 0.935 | 0.924 |
| 0.5 | Bicubic | 2883 | 225 | 497 | 0.928 | 0.853 | 0.889 |
| 0.5 | Real-ESRGAN | 2789 | 255 | 591 | 0.916 | 0.825 | 0.868 |

At matched thresholds Real-ESRGAN's ship recall is **2.0 pp below bicubic at conf 0.25 (0.881 vs 0.901) and 2.8 pp below at conf 0.50 (0.825 vs 0.853)** — the same direction as the ultralytics table but a far smaller gap than its mismatched operating points implied. SR also carries more unmatched detections on Set A (546 vs 439 at conf 0.25), consistent with the Set B invention result.

**Footnotes on this table.**

1. *"Unmatched detections" is not "false positives."* The original imagery itself produces **397 unmatched ship detections** at conf 0.25, against the same labels. DOTA's ship annotations are demonstrably incomplete — it omits small craft — so a detection with no matching label is not evidence that the detector was wrong. The column is named for what it measures: a detection that found no ground-truth box to pair with. The *relative* comparison across versions still holds, because all three are scored against the same labels.
2. *Ground truth here is 3380 ship instances across all 100 Set A tiles* (TP + Missed GT, identical for every version). That differs from the 3378 ultralytics reports, because ultralytics' validator drops `P0261__1024__390___0` — one of its DOTA polygon vertices sits at 1.179, outside the crop — while this fixed-threshold script reads the label files directly and keeps all 100 tiles.
3. *IoU is `batch_probiou` from ultralytics*, a Gaussian (probabilistic) approximation of rotated-box IoU rather than exact polygon intersection. It is applied identically to all three versions, so it cannot favour one of them; it does mean the absolute matching at the 0.5 threshold is approximate.


---
Generated by `scripts/06_summary.py` from `results/setA_metrics.json`, `results/setB_counts.csv`, `results/psnr.csv`, `results/setA_fixed_threshold.json`.
