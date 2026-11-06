"""Does the component check still catch damage? Paint defects onto real devices, rerun it on those types.

    uv run python scripts/dev/plant_components.py runs/<run>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch  # noqa: F401
import numpy as np
import zarr

from chipscan.io.readers import PyramidSource
from chipscan.stages.components import _type_key, component_errors, find_instances


class Planted(PyramidSource):
    def __init__(self, path, patches):
        super().__init__(path)
        self.patches = patches

    def read_region(self, x0, y0, x1, y1, level=0):
        a = super().read_region(x0, y0, x1, y1, level)
        if level:
            return a
        a = a.copy()
        for px, py, p in self.patches:
            h, w = p.shape
            ix0, iy0, ix1, iy1 = max(x0, px), max(y0, py), min(x1, px + w), min(y1, py + h)
            if ix0 < ix1 and iy0 < iy1:
                a[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = p[iy0 - py : iy1 - py, ix0 - px : ix1 - px]
        return a


if __name__ == "__main__":
    run = Path(sys.argv[1])
    meta = json.loads((run / "scan.json").read_text())
    px = meta["spec"]["pixel_size_um"]
    cls = zarr.open_group(str(run / "layout.zarr"), mode="r")["0"]
    insts = find_instances(cls)
    base = PyramidSource(run / "corrected.zarr")
    rng = np.random.default_rng(3)
    types = Counter(_type_key(d) for d in insts)
    planted, patches = [], []
    for key, n in types.most_common(12):
        if n < 30:
            continue
        members = [d for d in insts if _type_key(d) == key]
        for d in rng.choice(len(members), 3, replace=False):
            d = members[d]
            x0, y0, x1, y1 = d["bbox"]
            p = base.read_region(x0, y0, x1, y1).copy()
            h, w = p.shape
            kind = rng.integers(3)
            if kind == 0:  # a particle on the device
                r = max(2, min(h, w) // 6)
                yy, xx = np.mgrid[:h, :w]
                p[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r * r] = 40
            elif kind == 1:  # a missing piece of metal: a quarter of the device blanked to its background
                p[: h // 2, : w // 2] = int(np.median(base.read_region(x0 - 20, y0 - 20, x0 - 4, y0 - 4)))
            else:  # a bridge across the device
                p[h // 2 - 1 : h // 2 + 2, :] = 30
            patches.append((x0, y0, p))
            planted.append({"key": key, "kind": ["particle", "missing", "bridge"][kind], "bbox": d["bbox"]})
    src = Planted(run / "corrected.zarr", patches)
    keys = {p["key"] for p in planted}
    sub = [d for d in insts if _type_key(d) in keys]
    recs, summ = component_errors(src, sub, px, z_min=meta["spec"].get("component_z", 12.0), class_map=cls, report_missing=False)
    clean, _ = component_errors(base, sub, px, z_min=meta["spec"].get("component_z", 12.0), class_map=cls, report_missing=False)
    hit = 0
    for p in planted:
        x0, y0, x1, y1 = p["bbox"]
        ok = any(r["centroid_px"][0] >= x0 - 3 and r["centroid_px"][0] <= x1 + 3 and r["centroid_px"][1] >= y0 - 3 and r["centroid_px"][1] <= y1 + 3 for r in recs)
        hit += ok
        p["found"] = ok
    print(f"planted {len(planted)} on {len(keys)} device types: found {hit}; flags on the clean scan for these types: {len(clean)}")
    print(Counter((p["kind"], p["found"]) for p in planted))
