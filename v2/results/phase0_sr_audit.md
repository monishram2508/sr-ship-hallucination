# Phase 0: audit of the v1 super-resolution pipeline

**Verdict: no bug in the SR pipeline.** v1's Real-ESRGAN output is bit-identical to the official inference code run on the same device, and v1's degraded and bicubic images recompute bit-for-bit from the originals. Greyscale tiles do come out of Real-ESRGAN with colour in them; that is the model, not the code, and section 2 measures it (v1b). **One real bug was found, in v1's Set A evaluation rather than the SR path:** the detector ran with ultralytics' default `max_det=300`, which truncated two dense tiles (section 3).

## 1. The seven checks

| # | Check | How it was tested | Result |
|---|---|---|---|
| 1 | Channel order | `run_sr` reverses BGR to RGB going in and back coming out (2 flips in the source, quoted below); compared with the official code, which does the same with `cvtColor` | bit-identical to the official output on all 5 reference tiles; a deliberately swapped run differs by 40.8 dB, 35.9 dB, 34.5 dB on the colour tiles, so a swap would have shown |
| 2 | Range, dtype, mode | parameters, `eval()`, `no_grad`, the tensor conversion in `run_sr` | float in [0, 1], NCHW, output clamped to [0, 1] then `*255`, rounded, uint8; parameters torch.float32 (no fp16 anywhere); eval mode True; `torch.no_grad()` True |
| 3 | Greyscale tiles | channel equality in every version | passed in as three identical channels (LR stays grey on 92 of 92 grey tiles); SR output is grey on 0 of them (section 2) |
| 4 | Tiling seams | source | none possible: v1 sends each whole 256x256 LR tile through the model in one call |
| 5 | Degradation | v1's LR and bicubic PNGs recomputed from HR with v1's own functions; OpenCV's default kernel for sigma 2 vs an explicit 13x13 (= 6 sigma + 1) and a 25x25 kernel | LR exact on 250/250 tiles, bicubic exact on 250/250; the default kernel equals 13x13 on 250/250; a 25x25 kernel changes the LR by at most 0 grey levels; one LR array feeds both arms; every version is PNG |
| 6 | Official reference | `realesrgan` 0.3.0 `RealESRGANer`, fp32, CPU, tile 0, on 5 Set B LR tiles (2 grey, 3 colour) | see table below |
| 7 | Visual grid | `v2/results/sanity_grid.png` (local only: DOTA-derived) | 8 tiles incl. P1751: original, LR, bicubic, SR |

Upstream of v1, the originals are JPEG: ultralytics ships DOTA as `.jpg` and its tiler writes JPEG crops. v1's HR PNGs equal those JPEG tiles exactly (250/250), so JPEG artefacts are in the originals, identically for every arm, and the sigma-2 blur plus 4x area reduction removes most of them before either restoration sees the image.

### Reference comparison (PSNR, higher = closer; 'identical' = every pixel equal)

| tile | grey | official (pre_pad 0) vs v1 on CPU | official vs v1's file (MPS) | v1 MPS vs CPU | official pre_pad 10 vs v1 | channel-swapped vs official | border / interior diff, pre_pad 0 vs 10 |
|---|---|---|---|---|---|---|---|
| P1751__1024__0___2976 | yes | identical | 89.8 dB | 89.8 dB | 40.2 dB | 46.3 dB | 1.81 / 0.33 levels |
| P1610__1024__1648___1648 | yes | identical | 90.2 dB | 90.2 dB | 46.9 dB | 47.5 dB | 1.13 / 0.32 levels |
| P2256__1024__824___506 | no | identical | 92.2 dB | 92.2 dB | 51.3 dB | 40.8 dB | 0.31 / 0.07 levels |
| P0579__1024__1116___0 | no | identical | 91.1 dB | 91.1 dB | 44.1 dB | 35.9 dB | 1.56 / 0.20 levels |
| P1022__1024__0___79 | no | identical | 90.0 dB | 90.0 dB | 42.7 dB | 34.5 dB | 1.58 / 0.23 levels |

- **Implementation:** v1's spandrel path and the official code produce the same bytes on CPU. v1's stored outputs were made on Apple's GPU (MPS) and differ from CPU by float rounding only (90-92 dB).
- **pre_pad:** the official command-line script defaults to `pre_pad=0`, which v1 matches exactly. The class default `pre_pad=10` reflect-pads the input first, which changes the output (40-51 dB), mostly near the tile edge (border vs interior column). That is a different, equally official configuration, not a bug.

## 2. Greyscale tiles and v1b

67 of 150 Set B tiles and 25 of 100 Set A tiles are greyscale. Real-ESRGAN writes colour into every one of them: mean channel difference 1.24 grey levels on Set B (max 36), against exactly 0 for the original, LR and bicubic. The official code does the same for a 3-channel file; it converts back to grey only when the input file has one channel. So v1 followed the official behaviour, but a careful pipeline could reasonably convert grey tiles back, so the effect is measured.

**Where v1's Set B firing happened** (v1's own results, split by tile colour):

| version | conf | grey tiles fired | colour tiles fired | Fisher p |
|---|---|---|---|---|
| sr | 0.25 | 5/67 (7.5%) | 38/83 (45.8%) | 1.5e-07 |
| bicubic | 0.25 | 0/67 (0.0%) | 9/83 (10.8%) | 0.0045 |
| sr | 0.5 | 2/67 (3.0%) | 12/83 (14.5%) | 0.022 |
| bicubic | 0.5 | 0/67 (0.0%) | 2/83 (2.4%) | 0.5 |

SR's firing sits overwhelmingly on colour tiles, so false colour on grey tiles cannot be what drives v1's result. P1751, the dominant tile, is one of the few grey tiles that fired.

**v1 vs v1b** (the v1 detector, unchanged; v1b = SR with grey tiles converted back to grey):

| metric | conf | v1 SR | v1b SR | bicubic |
|---|---|---|---|---|
| Set B firing (tiles) | 0.25 | 43/150 (28.7%) | 45/150 (30.0%) | 9/150 (6.0%) |
| Set B firing (tiles) | 0.5 | 14/150 (9.3%) | 13/150 (8.7%) | 2/150 (1.3%) |
| Set A recall | 0.25 | 0.907 | 0.904 | 0.928 |
| Set A precision | 0.25 | 0.842 | 0.842 | 0.871 |
| Set A recall | 0.5 | 0.849 | 0.846 | 0.878 |
| Set A precision | 0.5 | 0.915 | 0.915 | 0.925 |

Invention for v1b needs verdicts on any box v1 never audited; those boxes are in the v2 audit pool (Phase 6), so v1b invention is reported with the v2 results.

## 3. A bug outside the SR path: v1's Set A evaluation was truncated

v2's evaluation is gated on reproducing v1's numbers. Predicting at conf 0.01 and thresholding afterwards first gave *more* Set A detections than v1's direct runs. The cause: v1 called the detector with ultralytics' default `max_det=300` (all 15 classes count toward it), and 2 dense harbour tiles (P0838__1024__0___0, P0897__1024__1307___1648) hit that cap on every version, so ship boxes were silently dropped. With v1's exact call the v2 code reproduces v1's TP/FP/FN exactly; with `max_det=1000` no tile reaches the cap.

| version | conf | ship boxes cut | v1 recall (as published) | corrected recall | v1 precision | corrected precision |
|---|---|---|---|---|---|---|
| hr | 0.25 | 123 | 0.951 | 0.978 | 0.890 | 0.885 |
| hr | 0.5 | 109 | 0.935 | 0.960 | 0.913 | 0.909 |
| bicubic | 0.25 | 119 | 0.901 | 0.928 | 0.874 | 0.871 |
| bicubic | 0.5 | 100 | 0.853 | 0.878 | 0.928 | 0.925 |
| sr | 0.25 | 114 | 0.881 | 0.907 | 0.845 | 0.842 |
| sr | 0.5 | 92 | 0.825 | 0.849 | 0.916 | 0.915 |

Every version loses the same two tiles, so absolute recall was understated but the comparison barely moves: SR minus bicubic recall at 0.25 is -0.020 as published and -0.022 corrected. Set B is unaffected (empty harbours never approach 300 boxes; v2 reproduces all 900 of v1's Set B per-tile counts exactly). v1's published files are left unchanged, as the v2 rules require; the corrected numbers live in `v2/results/setA_metrics.json` (`v1_offshelf`).

## 4. Consequence for v2

No bug in the SR path, so no image is regenerated: v2 imports v1's `degrade()` and `run_sr()` unchanged and runs them on the training tiles too. v1b is reported as a sensitivity row next to v1 throughout. Every v2 prediction uses `max_det=1000` and records any image that reaches it.

## Appendix: the v1 functions under test

```python
def degrade(hr):
    """HR uint8 BGR 1024 -> LR uint8 BGR 256. Shermeyer & Van Etten (2019) sensor model:
    Gaussian PSF blur then inter-area decimation."""
    blurred = cv2.GaussianBlur(hr, (0, 0), sigmaX=BLUR_SIGMA)
    return cv2.resize(blurred, (LR, LR), interpolation=cv2.INTER_AREA)

def run_sr(model, lr_bgr, dev):
    x = torch.from_numpy(lr_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(dev)
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    out = (y[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).round().astype("uint8")
    del x, y
    return out
```
