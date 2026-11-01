"""Phase 6: the blind labeller, which is v1's 11_label.py pointed at v2's crops.

Same window, same keys (s ship / n not_ship / u unsure / b back / q quit), saves after every
keypress and resumes at the first unlabelled crop. v1's script is imported unchanged; only its two
path constants are redirected to v2/results/audit/. It never opens audit_key.csv, so the labeller
cannot tell a new crop from a hidden repeat.

--queue NAME (added after first results) labels only the still-unlabelled crops of the
(condition, version) cells listed under audit.queues.NAME in config.yaml. The queue is written once
to v2/results/audit/queue_NAME.csv, in audit_labels.csv order, and its verdicts are merged into the
blanks of audit_labels.csv whenever the labeller exits, so quitting never loses work.

--recheck (added after first results) is a blind test-retest pass: config audit.recheck picks
n_primary crops of the priority cells at random, stratified by their previous verdict, plus every crop
of the originals queue, shuffles them and writes a blank v2/results/audit/audit_labels_recheck.csv.
The previous verdicts go to audit_recheck_key.csv, which the labeller never opens, and the re-check
rule is printed under every crop. Nothing is merged: 13_recheck.py compares the two passes.
"""
import argparse, csv, random, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_RESULTS, load_v1_script  # noqa: E402

AUD = V2_RESULTS / "audit"
RECHECK = AUD / "audit_labels_recheck.csv"
RECHECK_KEY = AUD / "audit_recheck_key.csv"
VERDICTS = ("ship", "unsure", "not_ship")


def queue_ids(m11, cells):
    """-> unlabelled crop ids whose cluster has a detection in one of `cells`, in labels-file order."""
    ids, labs = m11.read(AUD / "audit_labels.csv")
    clusters = {r["cluster"]: r for r in csv.DictReader(open(AUD / "clusters.csv"))}
    want = {clusters[d["cluster"]]["crop_id"] for d in csv.DictReader(open(AUD / "detections.csv"))
            if (d["condition"], d["version"]) in cells and not clusters[d["cluster"]]["verdict"]}
    return [c for c in ids if c in want and not labs[c]]


def allocate(n, sizes):
    """Split n equally across strata, capped at each stratum's size; leftovers go to the others."""
    take, left = dict.fromkeys(sizes, 0), n
    open_ = [k for k in sizes if sizes[k]]
    while left and open_:
        share = max(1, left // len(open_))
        for k in list(open_):
            add = min(share, sizes[k] - take[k], left)
            take[k], left = take[k] + add, left - add
            if take[k] == sizes[k]:
                open_.remove(k)
    return take


def primary_strata(m11):
    """-> {first-pass verdict: sorted crop ids} for the new crops of the priority cells."""
    _, labs = m11.read(AUD / "audit_labels.csv")
    clusters = {r["cluster"]: r for r in csv.DictReader(open(AUD / "clusters.csv"))}
    cells = {tuple(c) for c in CFG["audit"]["priority_cells"]}
    primary = sorted({clusters[d["cluster"]]["crop_id"] for d in csv.DictReader(open(AUD / "detections.csv"))
                      if (d["condition"], d["version"]) in cells and not clusters[d["cluster"]]["verdict"]})
    strata = {v: [c for c in primary if labs[c] == v] for v in VERDICTS}
    if sum(map(len, strata.values())) != len(primary):
        sys.exit("a priority-cell crop has no valid verdict; finish the main labelling first")
    return strata


def recheck_sample(m11):
    """-> shuffled [(crop_id, source, previous verdict)] for the blind re-check."""
    rc = CFG["audit"]["recheck"]
    _, labs = m11.read(AUD / "audit_labels.csv")
    strata = primary_strata(m11)
    rng = random.Random(rc["seed"])
    take = allocate(rc["n_primary"], {v: len(s) for v, s in strata.items()})
    picked = [(c, "primary", v) for v, s in strata.items() for c in rng.sample(s, take[v])]
    _, queue = m11.read(AUD / f"queue_{rc['queue']}.csv")
    picked += [(c, "originals_queue", labs[c]) for c in queue]
    rng.shuffle(picked)
    return picked


def with_rule(canvas, rule):
    """Wrap v1's canvas so the re-check rule is printed under every crop."""
    def draw(*args):
        c = canvas(*args)
        bar = np.full((44, c.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, rule, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (90, 210, 255), 2, cv2.LINE_AA)
        return np.vstack([c, bar])
    return draw


def merge(m11, queue):
    """Copy the queue's verdicts into the blanks of audit_labels.csv; a clash is a hard error."""
    ids, labs = m11.read(AUD / "audit_labels.csv")
    _, new = m11.read(queue)
    clash = [c for c, v in new.items() if v and labs[c] and labs[c] != v]
    if clash:
        sys.exit(f"queue verdicts disagree with audit_labels.csv for {clash[:5]}; nothing merged")
    m11.save(ids, {**labs, **{c: v for c, v in new.items() if v}}, AUD / "audit_labels.csv")
    print(f"merged {sum(1 for v in new.values() if v)} of {len(new)} queue verdicts into audit_labels.csv")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--queue", choices=sorted(CFG["audit"].get("queues", {})))
    mode.add_argument("--recheck", action="store_true")
    args = ap.parse_args()
    if not (AUD / "audit_labels.csv").exists():
        sys.exit("no v2 crops yet: run v2/scripts/07_build_audit_crops.py first")
    m11 = load_v1_script("11_label")
    m11.LABELS = AUD / "audit_labels.csv"
    m11.CROPS = AUD / "crops"
    if args.recheck:
        if not RECHECK.exists():
            picked = recheck_sample(m11)
            with open(RECHECK_KEY, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["crop_id", "source", "old_verdict"])
                w.writerows(picked)
            m11.save([c for c, _, _ in picked], {}, RECHECK)
            print(f"re-check: {len(picked)} crops -> {RECHECK}")
        m11.LABELS, m11.canvas = RECHECK, with_rule(m11.canvas, CFG["audit"]["recheck"]["rule"])
        sys.argv = [sys.argv[0]]
        m11.main()
        print("when every crop is re-labelled: .venv/bin/python v2/scripts/13_recheck.py")
        return
    queue = AUD / f"queue_{args.queue}.csv" if args.queue else None
    if queue and not queue.exists():
        ids = queue_ids(m11, {tuple(c) for c in CFG["audit"]["queues"][args.queue]})
        m11.save(ids, {}, queue)
        print(f"queue {args.queue}: {len(ids)} crops -> {queue}")
    if queue:
        m11.LABELS = queue
    sys.argv = [sys.argv[0]]          # v1's --pass2 mode is not used in v2
    try:
        m11.main()
    finally:
        if queue:
            merge(m11, queue)
    print("when every crop is labelled: .venv/bin/python v2/scripts/09_recount_invention.py")


if __name__ == "__main__":
    main()
