"""Evaluate the design-free anomaly detector on images with injected ground truth.

    uv run python scripts/eval_anomaly.py --model models/anomaly_v1.pt

Truth files store defect centres in design um plus the world->px transform, so they
are mapped to px here; the detector itself never sees a design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from chipscan.evaluate import score
from chipscan.geometry import Affine2D
from chipscan.io import open_source
from chipscan.segmentation.anomaly import AnomalyDetector, anomalies_from_map, write_anomaly_outputs

console = Console()

CASES = [
    ("synthetic_cpw", "data/synthetic/image.png", "data/synthetic/truth.json"),
    ("nft1_chip", "data/nft1_chip/image.zarr", "data/nft1_chip/truth.json"),
    ("bench_102", "data/benchmark/case_102/image.png", "data/benchmark/case_102/truth.json"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area-px", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("outputs/anomaly_eval"))
    a = ap.parse_args()
    det = AnomalyDetector(a.model)
    t = Table("case", "truth", "found", "false alarms", "recall", "precision", title=f"design-free detector {a.model.name}")
    results = {}
    for name, img, truth in CASES:
        if not Path(img).exists():
            continue
        src = open_source(img)
        g = zarr.open_group(str(a.out / f"{name}.anomaly.zarr"), mode="w")
        store = g.create_array("0", shape=src.shape, chunks=(1024, 1024), dtype="uint8")
        det.run(src, store=store, progress=False)
        recs = anomalies_from_map(store, a.threshold, a.min_area_px, None)
        write_anomaly_outputs(a.out, name, recs, src, store, None)
        tr = json.loads(Path(truth).read_text())
        w2p = Affine2D.from_dict(tr["world_to_px"])
        truth_px = []
        for d in tr["defects"]:
            px = w2p.apply(np.array([[d["x_um"], d["y_um"]]]))[0]
            truth_px.append({"class": "anomaly", "x_um": float(px[0]), "y_um": float(px[1]), "size_um": d["size_um"] * w2p.scale_factor})
        # detections already carry centroid_um == px when no pixel size is given
        sc = score(recs, truth_px, radius_um=8.0)
        results[name] = sc.to_dict()
        t.add_row(name, str(sc.n_truth), str(sc.found), str(sc.false_alarms), f"{sc.recall:.0%}", f"{sc.precision:.0%}")
    console.print(t)
    (a.out / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
