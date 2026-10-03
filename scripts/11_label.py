"""Blind labeller for the Set B box audit.

One crop at a time, full size. Keys:

    s = ship        a real vessel is there in the HR crop
    n = not_ship    rooftop, car, dock, wave, empty water -- no vessel
    u = unsure
    b = back one
    q = quit (or ESC, or closing the window)

Writes results/audit_labels.csv after EVERY keypress, so quitting never loses work, and resumes at
the first unlabelled crop. It reads only crop_id and the crop image: version, tile and confidence
live in results/audit_key.csv, which this script never opens.
"""
import csv, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, FIGURES  # noqa: E402

LABELS = RESULTS / "audit_labels.csv"
CROPS = FIGURES / "audit" / "crops"
LABELS2 = RESULTS / "audit_labels_pass2.csv"
CROPS2 = FIGURES / "audit" / "crops_pass2"
WIN = "Set B box audit"
KEYMAP = {ord("s"): "ship", ord("n"): "not_ship", ord("u"): "unsure"}
COLOUR = {"ship": (90, 200, 90), "not_ship": (60, 60, 230), "unsure": (40, 190, 230)}
HEAD, FOOT = 86, 56


def read(labels):
    rows = list(csv.DictReader(open(labels)))
    return [r["crop_id"] for r in rows], {r["crop_id"]: (r.get("verdict") or "").strip() for r in rows}


def save(ids, verdicts, labels):
    tmp = labels.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "verdict"])
        for c in ids:
            w.writerow([c, verdicts.get(c, "")])
    tmp.replace(labels)


def canvas(cid, img, i, n, verdict, n_done):
    h, w = img.shape[:2]
    c = np.full((h + HEAD + FOOT, w, 3), 24, np.uint8)
    c[HEAD:HEAD + h] = img
    cv2.putText(c, f"{i + 1} / {n}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(c, cid, (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
    if verdict:
        cv2.putText(c, verdict, (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    COLOUR.get(verdict, (255, 255, 255)), 2, cv2.LINE_AA)
    bx0, bx1, by = 16, w - 16, 62
    cv2.rectangle(c, (bx0, by), (bx1, by + 10), (70, 70, 70), -1)
    if n:
        cv2.rectangle(c, (bx0, by), (bx0 + int((bx1 - bx0) * n_done / n), by + 10), (90, 200, 90), -1)
    cv2.putText(c, "s ship    n not_ship    u unsure    b back    q quit",
                (16, h + HEAD + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (210, 210, 210), 2, cv2.LINE_AA)
    return c


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass2", action="store_true",
                    help="adjudicate only the crops marked `unsure` in pass 1, with wider context")
    args = ap.parse_args()
    labels, crops = (LABELS2, CROPS2) if args.pass2 else (LABELS, CROPS)
    if not labels.exists():
        sys.exit(f"{labels} not found. Run "
                 f"{'scripts/13_pass2.py' if args.pass2 else 'scripts/08_audit_boxes.py'} first.")
    if args.pass2:
        print("PASS 2 — re-adjudicating pass-1 `unsure` crops at 128 px context")
    ids, verdicts = read(labels)
    n = len(ids)
    todo = [k for k, c in enumerate(ids) if not verdicts[c]]
    if not todo:
        nxt = "scripts/09_recount.py" if not args.pass2 else "the merge step (tell Claude 'done')"
        print(f"All {n} crops already labelled. Next: {nxt}")
        return
    i = todo[0]
    print(f"{n - len(todo)}/{n} already done, resuming at {ids[i]}")
    print("keys: s=ship  n=not_ship  u=unsure  b=back  q=quit")

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cid = ids[i]
            img = cv2.imread(str(crops / f"{cid}.png"), cv2.IMREAD_COLOR)
            if img is None:
                sys.exit(f"missing crop image {crops/f'{cid}.png'}")
            n_done = sum(1 for c in ids if verdicts[c])
            cv2.imshow(WIN, canvas(cid, img, i, n, verdicts[cid], n_done))

            k = -1
            while k == -1:
                k = cv2.waitKey(30)
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    print("window closed")
                    return
            k &= 0xFF
            if k in (ord("q"), 27):
                print("quit")
                return
            if k == ord("b"):
                i = max(0, i - 1)
                continue
            if k in KEYMAP:
                verdicts[cid] = KEYMAP[k]
                save(ids, verdicts, labels)
                i += 1
                if i >= n:
                    left = [c for c in ids if not verdicts[c]]
                    if left:
                        i = ids.index(left[0])
                        print(f"wrapped to first unlabelled: {ids[i]}")
                        continue
                    print(f"\nall {n} crops labelled -> {labels}")
                    print("next: " + ("tell Claude 'done' to merge" if args.pass2
                                      else ".venv/bin/python scripts/09_recount.py"))
                    return
    finally:
        save(ids, verdicts, labels)
        cv2.destroyAllWindows()
        done = sum(1 for c in ids if verdicts[c])
        print(f"saved {done}/{n} verdicts to {labels}")


if __name__ == "__main__":
    main()
