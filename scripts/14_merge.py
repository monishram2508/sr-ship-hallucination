"""Merge pass-2 adjudication into the working labels.

Rule: a pass-2 verdict replaces the pass-1 verdict for that crop, and nothing else changes.
Pass 2 only ever contained crops that were `unsure` in pass 1, so this is asserted rather than
assumed: any attempt to alter a crop that was not `unsure` is a hard error.

results/audit_labels_pass1.csv stays read-only and is the record of the first pass.
"""
import csv, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

P1 = RESULTS / "audit_labels_pass1.csv"
P2 = RESULTS / "audit_labels_pass2.csv"
OUT = RESULTS / "audit_labels.csv"
VALID = {"ship", "not_ship", "unsure"}


def main():
    p1 = {r["crop_id"]: r["verdict"].strip() for r in csv.DictReader(open(P1))}
    p2 = {r["crop_id"]: r["verdict"].strip() for r in csv.DictReader(open(P2))}

    bad = [c for c, v in p2.items() if v not in VALID]
    if bad:
        sys.exit(f"invalid pass-2 verdicts: {bad[:10]}")
    blank = [c for c, v in p2.items() if not v]
    if blank:
        sys.exit(f"pass 2 incomplete: {len(blank)} blank ({blank[:10]})")
    wrong = [c for c in p2 if p1.get(c) != "unsure"]
    if wrong:
        sys.exit(f"pass 2 touches crops that were not `unsure` in pass 1: {wrong[:10]}")
    missing = [c for c, v in p1.items() if v == "unsure" and c not in p2]
    if missing:
        sys.exit(f"{len(missing)} pass-1 `unsure` crops absent from pass 2: {missing[:10]}")

    merged = dict(p1)
    flips = Counter()
    for c, v in p2.items():
        flips[f"unsure -> {v}"] += 1
        merged[c] = v

    order = [r["crop_id"] for r in csv.DictReader(open(P1))]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "verdict"])
        for c in order:
            w.writerow([c, merged[c]])

    unchanged = sum(1 for c in order if merged[c] == p1[c])
    print(f"merged {len(p2)} pass-2 verdicts into {len(order)} crops; "
          f"{unchanged} unchanged, {len(order)-unchanged} replaced")
    print("\npass-2 flips:")
    for k, v in sorted(flips.items()):
        print(f"  {k:22s} {v}")
    print("\ntally  pass1 -> final:")
    t1, t2 = Counter(p1.values()), Counter(merged.values())
    for k in ("ship", "not_ship", "unsure"):
        print(f"  {k:9s} {t1[k]:3d} -> {t2[k]:3d}   ({t2[k]-t1[k]:+d})")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
