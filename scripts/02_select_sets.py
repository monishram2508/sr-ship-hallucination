"""Phase 1b — build Set A (recovery) and Set B (invention).

Set A: 100 random tiles (seed 0) with >=1 ship LABEL.
Set B: tiles with a harbor(7) or bridge(8) label and ZERO ship labels, which the
       detector then confirms as empty — zero ship DETECTIONS on the HR tile at
       conf 0.25. Cap 150. Any ship that later appears on bicubic or SR was
       therefore introduced by degradation or restoration, not missed by DOTA.
"""
import csv, random, sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, DETECTOR, IMGSZ, SHIP_CLS, HARBOR_CLS, BRIDGE_CLS, device  # noqa: E402

TILES = DATA / "tiles"
SETS = DATA / "sets"
SEED = 0
N_A = 100
CAP_B = 150
MIN_B = 30


def label_classes(p: Path) -> Counter:
    c = Counter()
    txt = p.read_text().strip()
    if not txt:
        return c
    for line in txt.splitlines():
        parts = line.split()
        if parts:
            c[int(parts[0])] += 1
    return c


def main():
    SETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    img_by_stem = {p.stem: p for p in sorted((TILES / "images").glob("*"))}
    lbls = sorted((TILES / "labels").glob("*.txt"))
    print(f"tiles: {len(img_by_stem)} images, {len(lbls)} labels")

    classes = {p.stem: label_classes(p) for p in lbls}

    with_ship = [s for s, c in classes.items() if c[SHIP_CLS] > 0 and s in img_by_stem]
    no_ship = [s for s, c in classes.items() if c[SHIP_CLS] == 0 and s in img_by_stem]
    harbor_bridge_no_ship = [s for s in no_ship if classes[s][HARBOR_CLS] > 0 or classes[s][BRIDGE_CLS] > 0]
    print(f"tiles with >=1 ship label      : {len(with_ship)}")
    print(f"tiles with 0 ship labels       : {len(no_ship)}")
    print(f"  of which harbor/bridge present: {len(harbor_bridge_no_ship)}")

    # ---- Set A -----------------------------------------------------------
    rng = random.Random(SEED)
    setA = sorted(rng.sample(with_ship, min(N_A, len(with_ship))))
    (SETS / "setA.txt").write_text("\n".join(setA) + "\n")
    n_ship_labels = sum(classes[s][SHIP_CLS] for s in setA)
    print(f"Set A: {len(setA)} tiles, {n_ship_labels} ship labels "
          f"(min {min(classes[s][SHIP_CLS] for s in setA)}, max {max(classes[s][SHIP_CLS] for s in setA)})")

    # ---- Set B -----------------------------------------------------------
    from ultralytics import YOLO
    dev = device()
    model = YOLO(DETECTOR)

    candidates = harbor_bridge_no_ship
    relaxed = False
    verified, rejected = [], []

    CHUNK = 8  # a list source makes ultralytics build one tensor per call; 608 x 1024px = 7 GiB

    def verify(cands):
        keep, rej = [], []
        with tqdm(total=len(cands), desc="verify-empty on HR") as bar:
            for i in range(0, len(cands), CHUNK):
                group = cands[i:i + CHUNK]
                paths = [str(img_by_stem[s]) for s in group]
                for stem, r in zip(group, model.predict(paths, imgsz=IMGSZ, conf=0.25,
                                                        device=dev, verbose=False)):
                    n = 0 if r.obb is None else int((r.obb.cls == SHIP_CLS).sum())
                    (keep if n == 0 else rej).append((stem, n))
                bar.update(len(group))
        return keep, rej

    rng_b = random.Random(SEED)
    rng_b.shuffle(candidates)
    verified, rejected = verify(candidates)
    print(f"harbor/bridge candidates: {len(candidates)} -> verified empty {len(verified)}, "
          f"rejected (unlabelled boats) {len(rejected)}")

    if len(verified) < MIN_B:
        relaxed = True
        print(f"!! verified pool {len(verified)} < {MIN_B}: RELAXING to any tile with 0 ship labels")
        extra = [s for s in no_ship if s not in set(candidates)]
        rng_b.shuffle(extra)
        extra = extra[: CAP_B * 3]
        v2, r2 = verify(extra)
        verified += v2
        rejected += r2

    setB = sorted(s for s, _ in verified[:CAP_B])
    (SETS / "setB.txt").write_text("\n".join(setB) + "\n")

    with open(RESULTS / "setB_verification.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "hr_ship_detections_conf0.25", "kept"])
        for s, n in verified:
            w.writerow([s, n, int(s in set(setB))])
        for s, n in rejected:
            w.writerow([s, n, 0])

    print(f"Set B: {len(setB)} tiles (cap {CAP_B}){' [RELAXED]' if relaxed else ''}")
    print(f"rejected {len(rejected)} tiles that had unlabelled boats visible to the detector")
    print(f"-> {SETS/'setA.txt'}  {SETS/'setB.txt'}  {RESULTS/'setB_verification.csv'}")


if __name__ == "__main__":
    main()
