"""Phase 6: the blind labeller, which is v1's 11_label.py pointed at v2's crops.

Same window, same keys (s ship / n not_ship / u unsure / b back / q quit), saves after every
keypress and resumes at the first unlabelled crop. v1's script is imported unchanged; only its two
path constants are redirected to v2/results/audit/. It never opens audit_key.csv, so the labeller
cannot tell a new crop from a hidden repeat.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import V2_RESULTS, load_v1_script  # noqa: E402

AUD = V2_RESULTS / "audit"


def main():
    if not (AUD / "audit_labels.csv").exists():
        sys.exit("no v2 crops yet: run v2/scripts/07_build_audit_crops.py first")
    m11 = load_v1_script("11_label")
    m11.LABELS = AUD / "audit_labels.csv"
    m11.CROPS = AUD / "crops"
    sys.argv = [sys.argv[0]]          # v1's --pass2 mode is not used in v2
    m11.main()
    print("when every crop is labelled: .venv/bin/python v2/scripts/09_recount_invention.py")


if __name__ == "__main__":
    main()
