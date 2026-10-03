"""QA contact sheets: every crop grouped by the verdict it was given.

One sheet per verdict so a labelling pass can be eyeballed for consistency. Still blind: crop_id
only, no version, tile or confidence. Reads audit_labels.csv and the crop images; never opens
audit_key.csv.
"""
import csv, math, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, FIGURES  # noqa: E402

AUDIT = FIGURES / "audit"
CROPS = AUDIT / "crops"
CAP = 24
GAP = 8


def sheet(ids, verdict, out):
    n = len(ids)
    if n == 0:
        print(f"no crops labelled {verdict}")
        return
    cell = 260 if n <= 64 else 220
    cols = min(10, max(4, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    W = cols * cell + (cols + 1) * GAP
    H = rows * (cell + CAP) + (rows + 1) * GAP + 46
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.putText(img, f"QA sheet - verdict '{verdict}'  -  {n} crops", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for k, cid in enumerate(ids):
        r, c = divmod(k, cols)
        src = cv2.imread(str(CROPS / f"{cid}.png"), cv2.IMREAD_COLOR)
        if src is None:
            continue
        thumb = cv2.resize(src, (cell, cell), interpolation=cv2.INTER_AREA)
        x = GAP + c * (cell + GAP)
        y = 46 + GAP + r * (cell + CAP + GAP)
        img[y:y + cell, x:x + cell] = thumb
        cv2.putText(img, cid, (x + 4, y + cell + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)
    print(f"{verdict:9s} {n:3d} crops -> {out}")


def main():
    labs = list(csv.DictReader(open(RESULTS / "audit_labels.csv")))
    for verdict in ("ship", "not_ship", "unsure"):
        ids = [r["crop_id"] for r in labs if (r.get("verdict") or "").strip() == verdict]
        sheet(ids, verdict, AUDIT / f"qa_{verdict}.png")


if __name__ == "__main__":
    main()
