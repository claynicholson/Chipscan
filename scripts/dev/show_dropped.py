"""Contact sheet of random dropped speck candidates for one reason: uv run python scripts/dev/show_dropped.py runs/<run> <reason> out.jpg"""
import json
import sys
from pathlib import Path

import torch  # noqa: F401
import cv2
import numpy as np

from chipscan.io.readers import PyramidSource

if __name__ == "__main__":
    run, reason, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    src = PyramidSource(run / "corrected.zarr")
    d = [x for x in json.load(open(run / "dropped.json")) if x["reason"] == reason]
    rng = np.random.default_rng(0)
    pick = [d[i] for i in rng.choice(len(d), min(48, len(d)), replace=False)]
    tiles = []
    for x in pick:
        b = x["bbox_px"]
        cx, cy = int((b["x0"] + b["x1"]) / 2), int((b["y0"] + b["y1"]) / 2)
        R = 32
        a = cv2.cvtColor(src.read_region(cx - R, cy - R, cx + R, cy + R), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(a, (int(b["x0"]) - cx + R - 2, int(b["y0"]) - cy + R - 2), (int(b["x1"]) - cx + R + 1, int(b["y1"]) - cy + R + 1), (0, 0, 255), 1)
        tiles.append(cv2.resize(a, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST))
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]))
