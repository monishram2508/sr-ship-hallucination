"""Phase 0 sensitivity ("v1b"): v1's SR outputs with greyscale tiles converted back to grey.

Real-ESRGAN writes colour into greyscale tiles (three identical channels in, three different
channels out). The official realesrgan code converts its output back to grey only when the input
file has one channel; for a 3-channel grey file it does exactly what v1 did. So this is not a bug
fix but a defensible alternative pipeline, and v1b measures whether it matters.

Writes v2/data/v1b/{setA,setB}/images/*.png: grey tiles -> cv2 BGR2GRAY of v1's SR (the luma
conversion the official code applies to 1-channel input), replicated to 3 channels; colour tiles
-> a symlink to v1's SR file, unchanged. 04_predict.py then runs every model on it as version
"sr_greyfix".
"""
import json, os, sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import V2_DATA, V2_RESULTS, SETS, read_ids, version_image, is_grey  # noqa: E402

OUT = V2_DATA / "v1b"


def main():
    summary = {}
    for s in SETS:
        d = OUT / s / "images"
        d.mkdir(parents=True, exist_ok=True)
        changed = []
        for t in tqdm(read_ids(s), desc=f"v1b {s}", leave=False):
            dst = d / f"{t}.png"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            hr = cv2.imread(str(version_image(s, "hr", t)), cv2.IMREAD_COLOR)
            src = version_image(s, "sr", t)
            if is_grey(hr):
                g = cv2.cvtColor(cv2.imread(str(src), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(dst), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
                changed.append(t)
            else:
                os.symlink(src.resolve(), dst)
        summary[s] = {"n_tiles": len(read_ids(s)), "n_grey_converted": len(changed)}
        print(f"{s}: {len(changed)} grey tiles converted, the rest linked to v1's SR")
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    (V2_RESULTS / "v1b_build.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {OUT}  (next: v2/scripts/04_predict.py, then 05 and 06)")


if __name__ == "__main__":
    main()
