"""Re-score existing runs with another speck model: speck map (all chip tiles), then findings.

    uv run python scripts/dev/apply_speck_model.py models/speck_v4.pt runs/AFT5_07_C1_1009_161048 ...
"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401

import zarr

from chipscan.io.readers import PyramidSource
from chipscan.stages.flow import _max_levels, _new_map, reprocess, scan_mask
from chipscan.stages.nets import TiledNet, run_nets

if __name__ == "__main__":
    model = Path(sys.argv[1]).resolve()
    for r in map(Path, sys.argv[2:]):
        t = time.time()
        meta = json.loads((r / "scan.json").read_text())
        src = PyramidSource(r / "corrected.zarr")
        g = zarr.open_group(str(r / "speck.zarr"), mode="r+")
        n = run_nets(src, [(TiledNet(model), g["0"])], mask=scan_mask(r, src, meta["spec"].get("pixel_size_um")))
        print(r, f"speck map: {n} tiles, {time.time() - t:.0f}s", flush=True)
        hg, heat = _new_map(r / "heat.zarr", src.shape)
        sp, tr = g["0"], zarr.open_group(str(r / "trace.zarr"), mode="r")["0"]
        H, W = src.shape
        import numpy as np

        for y in range(0, H, 2048):
            for x in range(0, W, 2048):
                heat[y : y + 2048, x : x + 2048] = np.maximum(np.asarray(sp[y : y + 2048, x : x + 2048]), np.asarray(tr[y : y + 2048, x : x + 2048]))
        _max_levels(hg)
        meta["spec"]["speck_model"] = str(model)
        (r / "scan.json").write_text(json.dumps(meta, indent=2, default=str))
        o = reprocess(r)
        print(r, json.dumps(o["summary"]), f"total {time.time() - t:.0f}s", flush=True)
