"""Phase 6: the blind labeller, which is v1's 11_label.py pointed at v2's crops.

Same window, same keys (s ship / n not_ship / u unsure / b back / q quit), saves after every
keypress and resumes at the first unlabelled crop. v1's script is imported unchanged; only its two
path constants are redirected to v2/results/audit/. It never opens audit_key.csv, so the labeller
cannot tell a new crop from a hidden repeat.

--queue NAME (added after first results) labels only the still-unlabelled crops of the
(condition, version) cells listed under audit.queues.NAME in config.yaml. The queue is written once
to v2/results/audit/queue_NAME.csv, in audit_labels.csv order, and its verdicts are merged into the
blanks of audit_labels.csv whenever the labeller exits, so quitting never loses work.
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import CFG, V2_RESULTS, load_v1_script  # noqa: E402

AUD = V2_RESULTS / "audit"


def queue_ids(m11, cells):
    """-> unlabelled crop ids whose cluster has a detection in one of `cells`, in labels-file order."""
    ids, labs = m11.read(AUD / "audit_labels.csv")
    clusters = {r["cluster"]: r for r in csv.DictReader(open(AUD / "clusters.csv"))}
    want = {clusters[d["cluster"]]["crop_id"] for d in csv.DictReader(open(AUD / "detections.csv"))
            if (d["condition"], d["version"]) in cells and not clusters[d["cluster"]]["verdict"]}
    return [c for c in ids if c in want and not labs[c]]


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
    ap.add_argument("--queue", choices=sorted(CFG["audit"].get("queues", {})))
    args = ap.parse_args()
    if not (AUD / "audit_labels.csv").exists():
        sys.exit("no v2 crops yet: run v2/scripts/07_build_audit_crops.py first")
    m11 = load_v1_script("11_label")
    m11.LABELS = AUD / "audit_labels.csv"
    m11.CROPS = AUD / "crops"
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
