"""Look-alike comparison for every device, swept over sensitivity: recall on planted damage vs flags on the clean scan."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401
import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from plant_components import Planted  # noqa: E402

from chipscan.io.readers import PyramidSource
from chipscan.stages.components import _ZarrRegion, _patch_size, _type_key, compare_group, find_instances, read_patches

if __name__ == "__main__":
    run = Path(sys.argv[1])
    cls = zarr.open_group(str(run / "layout.zarr"), mode="r")["0"]
    insts = find_instances(cls)
    base = PyramidSource(run / "corrected.zarr")
    rng = np.random.default_rng(3)
    by = defaultdict(list)
    for i, d in enumerate(insts):
        by[_type_key(d)].append(i)
    keys = [k for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(v) >= 30][:12]
    patches, planted = [], {}
    for key in keys:
        for j in rng.choice(len(by[key]), 3, replace=False):
            i = by[key][j]
            x0, y0, x1, y1 = insts[i]["bbox"]
            p = base.read_region(x0, y0, x1, y1).copy()
            h, w = p.shape
            kind = int(rng.integers(3))
            if kind == 0:
                r = max(2, min(h, w) // 6)
                yy, xx = np.mgrid[:h, :w]
                p[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r * r] = 40
            elif kind == 1:
                p[: h // 2, : w // 2] = int(np.median(base.read_region(x0 - 20, y0 - 20, x0 - 4, y0 - 4)))
            else:
                p[h // 2 - 1 : h // 2 + 2, :] = 30
            patches.append((x0, y0, p))
            planted[i] = ["particle", "missing", "bridge"][kind]
    src = Planted(run / "corrected.zarr", patches)
    res = {}
    for name, s in (("planted", src), ("clean", base)):
        rows = []  # per member: max z inside the footprint, ncc
        for key in keys:
            idx = by[key]
            size = _patch_size(key)
            centers = np.array([[insts[i]["cx"], insts[i]["cy"]] for i in idx])
            P = read_patches(s, centers, size)
            Mk = read_patches(_ZarrRegion(cls), centers, size)
            foot = ndi.binary_dilation(np.mean(np.isclose(Mk * 255.0, 2.0), axis=0) > 0.5, iterations=2)
            flat = P.reshape(len(P), -1)
            flat = flat - flat.mean(1, keepdims=True)
            flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
            for k, i in enumerate(idx):
                sims = flat @ flat[k]
                sims[k] = -1
                nn = [int(j) for j in np.argsort(sims)[::-1][:12] if sims[j] > 0.6]
                if len(nn) < 5:
                    rows.append((i, 0.0, 1.0))
                    continue
                r = compare_group(P[nn + [k]], z_min=1.0, footprint=foot)[-1]
                rows.append((i, r["z"] if r["mask"] is not None else 0.0, r["ncc"]))
        res[name] = rows
    clean_z = {i: (z, c) for i, z, c in res["clean"]}
    for zt in (4, 6, 9, 12, 20):
        for nt in (0.3, 0.5):
            fl = lambda z, c: z >= zt or c < nt  # noqa: E731
            hits = sum(fl(z, c) for i, z, c in res["planted"] if i in planted)
            fa = sum(fl(z, c) for i, (z, c) in clean_z.items())
            print(f"z>={zt:<3} or ncc<{nt}: found {hits}/{len(planted)}, flags on clean {fa} of {len(clean_z)} devices")
