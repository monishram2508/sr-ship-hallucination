"""Phase 0 reference: run the OFFICIAL Real-ESRGAN inference code (realesrgan.RealESRGANer) on v1's
LR tiles, so 00_sr_audit.py can compare it against v1's spandrel outputs.

Runs in a separate, throwaway environment that has `realesrgan` 0.3.0 and `basicsr` 1.4.2 (basicsr
needs its torchvision import patched; phase0_sr_audit.md says how). fp32 on CPU, no tiling, and both
pre_pad settings the official code uses: 0 (inference_realesrgan.py default) and 10 (class default).

Input is read with cv2.IMREAD_UNCHANGED exactly as inference_realesrgan.py reads it. For greyscale
tiles it is also fed as a 1-channel array, the only case where the official code converts its
output back to grey.
"""
import argparse, sys, time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, repo_path, version_image, is_grey  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="comma-separated Set B tile ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = [t for t in args.tiles.split(",") if t]
    weights = str(repo_path(CFG["v1"]["sr_weights"]))
    for pre_pad in (0, 10):
        # the RRDBNet that inference_realesrgan.py builds for RealESRGAN_x4plus
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=weights, model=net, tile=0, pre_pad=pre_pad,
                          half=False, device=torch.device("cpu"))
        for t in tiles:
            lr = cv2.imread(str(version_image("setB", "lr", t)), cv2.IMREAD_UNCHANGED)
            t0 = time.perf_counter()
            y, _ = up.enhance(lr, outscale=4)
            print(f"pre_pad={pre_pad} {t} {lr.shape}->{y.shape} {time.perf_counter() - t0:.1f}s")
            cv2.imwrite(str(out / f"official_prepad{pre_pad}__{t}.png"), y)
            if pre_pad == 0 and is_grey(lr):
                y1, _ = up.enhance(lr[:, :, 0].copy(), outscale=4)
                cv2.imwrite(str(out / f"official_1ch__{t}.png"), y1)
                print(f"  1-channel input -> output shape {y1.shape}")


if __name__ == "__main__":
    main()
