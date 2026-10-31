"""Added after first results: what a retrained detector calls a ship on the untouched originals.

The 24 highest-confidence M_HR detections (any seed, conf > 0.25) on original Set B tiles whose
audit verdict is not_ship, one per audit region, each cropped from the original tile with v1's crop
renderer (box in magenta, same padding as the audit crops). Verdicts come from 09's own functions,
so the gallery cannot disagree with the recount.

Writes v2/results/figures/mhr_false_ships_on_originals.png and
v2/results/mhr_false_ships_on_originals.json (tile, seed, confidence, region, crop window).
"""
import importlib.util, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_v2 import (V2, V2_RESULTS, load_predictions, load_v1_script, prediction_files,  # noqa: E402
                       version_image)

N, COLS, CELL, LABEL_H = 24, 6, 300, 34
OUT_PNG = V2_RESULTS / "figures" / "mhr_false_ships_on_originals.png"
OUT_JSON = V2_RESULTS / "mhr_false_ships_on_originals.json"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, V2 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def false_ships():
    """-> the N highest-confidence M_HR detections on originals judged not_ship, one per region."""
    m09 = load_script("09_recount_invention")
    cv, v1key, v1final, _ = m09.verdicts(True)
    best = {}
    for d in m09.detection_rows(cv, v1key, v1final, False):
        if (d["condition"], d["version"], d["verdict"]) != ("M_HR", "hr", "not_ship"):
            continue
        if d["cluster"] not in best or d["conf"] > best[d["cluster"]]["conf"]:
            best[d["cluster"]] = d
    return sorted(best.values(), key=lambda d: -d["conf"])[:N]


def with_polygons(dets):
    """Attach each detection's polygon from 04's predictions, checking rank and confidence agree."""
    preds = load_predictions(prediction_files(sorted({d["model"] for d in dets})))
    out = []
    for d in dets:
        _, poly, conf = preds[(d["model"], "setB", "hr")][d["tile"]]
        r = int(d["rank"])
        if not np.isclose(conf[r], d["conf"], atol=1e-6):
            sys.exit(f"rank {r} of {d['model']} {d['tile']} has conf {conf[r]}, detections.csv says {d['conf']}")
        out.append({**d, "poly": poly[r]})
    return out


def gallery(dets, m08):
    cells = []
    for d in dets:
        hr = cv2.imread(str(version_image("setB", "hr", d["tile"])), cv2.IMREAD_COLOR)
        crop = m08.render(hr, d, CELL, 2)
        label = np.full((LABEL_H, CELL, 3), 252, np.uint8)
        cv2.putText(label, f"conf {d['conf']:.2f}  seed {d['seed']}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (43, 43, 42), 1, cv2.LINE_AA)
        cells.append(np.vstack([crop, label]))
    blank = np.full_like(cells[0], 252)
    cells += [blank] * (-len(cells) % COLS)
    rows = [np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)]
    return np.vstack(rows)


def main():
    m08 = load_v1_script("08_audit_boxes")
    dets = with_polygons(false_ships())
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), gallery(dets, m08))
    rows = [{"tile": d["tile"], "model": d["model"], "seed": d["seed"], "conf": d["conf"], "cluster": d["cluster"],
             "crop_box": list(m08.crop_box(d))} for d in dets]
    OUT_JSON.write_text(json.dumps({"n": len(rows), "n_tiles": len({r["tile"] for r in rows}),
                                    "conf_range": [rows[-1]["conf"], rows[0]["conf"]], "detections": rows}, indent=2))
    print(f"{len(rows)} not_ship regions from {len({r['tile'] for r in rows})} tiles, "
          f"conf {rows[-1]['conf']:.2f}-{rows[0]['conf']:.2f}\n-> {OUT_PNG}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
