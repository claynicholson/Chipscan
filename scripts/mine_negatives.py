"""Hard-negative mining: where does a detector fire on the training scans?

Real defects are rare on the training scans; what a trace-error net flags there is mostly
designed structure it mistakes for a defect (corners, line ends, test blocks).  The centres
are saved and DefectTiles samples tiles around them, unlabelled, so fine-tuning teaches
"this is normal".

    uv run python scripts/mine_negatives.py --model models/trace_v1.pt --out models/trace_negatives.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch  # noqa: F401

from chipscan.io.readers import PyramidSource
from chipscan.segmentation.anomaly import anomalies_from_map
from chipscan.stages.nets import TiledNet

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ["AFT5_01_A2", "AFT5_04_B2", "AHQ1_B1_reticle", "PCB_boards_batch1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    net = TiledNet(a.model)
    res: dict[str, list] = {}
    for name in TRAIN:
        src = PyramidSource(ROOT / "data/pyramids" / f"{name}.zarr")
        out = np.zeros(src.shape, np.uint8)
        net.run(src, out)
        recs = anomalies_from_map(out, a.threshold, 4, None, merge_px=8)
        res[name] = [[round(r["centroid_px"][0]), round(r["centroid_px"][1])] for r in recs]
        print(name, len(recs), flush=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
