"""Harvest real specks from reference maps of the training scans (centres for focused sampling).

A speck is a reference-residual blob isolated on flat field (see chipscan.stages.nets.ref_dirt).
Writes models/speck_harvest.json: {scan: [[x, y], ...]}.

    uv run python scripts/harvest_specks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401
import numpy as np
import zarr
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chipscan.io.readers import PyramidSource  # noqa: E402
from chipscan.stages.nets import ref_dirt  # noqa: E402

TRAIN = ["AFT5_01_A2", "AFT5_04_B2", "AHQ1_B1_reticle", "PCB_boards_batch1"]


def harvest(src: PyramidSource, ref, block: int = 2048) -> list[list[int]]:
    H, W = ref.shape
    out = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            r = np.asarray(ref[y : y + block, x : x + block])
            if (r > 60).sum() < 3:
                continue
            img = src.read_region(x, y, x + r.shape[1], y + r.shape[0]).astype(np.float32) / 255.0
            pos, _ = ref_dirt(img, r)
            lab, n = ndi.label(pos)
            for c in ndi.center_of_mass(pos, lab, range(1, n + 1)) if n else []:
                out.append([int(x + c[1]), int(y + c[0])])
    return out


def main() -> None:
    res = {}
    for n in TRAIN:
        pth = next((p for p in (ROOT / "data/refmaps" / f"{n}.reference.zarr", ROOT / "runs" / n / "reference.zarr") if p.exists()), None)
        if pth is None:
            continue
        src = PyramidSource(pth.parent / "corrected.zarr" if pth.parent.parent.name == "runs" else ROOT / "data/pyramids" / f"{n}.zarr")
        res[n] = harvest(src, zarr.open_group(str(pth), mode="r")["0"])
        print(n, len(res[n]), flush=True)
    (ROOT / "models" / "speck_harvest.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
