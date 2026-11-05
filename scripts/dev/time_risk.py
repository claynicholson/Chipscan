"""Time the parts of the risk stage on a finished run: uv run python scripts/dev/time_risk.py runs/<run>"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401
import zarr

from chipscan.io.readers import PyramidSource
from chipscan.segmentation.anomaly import anomalies_from_map
from chipscan.stages.flow import ScanSpec, check_candidates, speck_union_map
from chipscan.stages.risk import assess

if __name__ == "__main__":
    run = Path(sys.argv[1])
    meta = json.loads((run / "scan.json").read_text())
    spec = ScanSpec(**{k: v for k, v in meta["spec"].items() if k in ScanSpec.__dataclass_fields__})
    src = PyramidSource(run / "corrected.zarr")
    px = spec.pixel_size_um
    t = time.time()
    amap = speck_union_map(run, spec)
    print(f"union map {time.time() - t:.0f}s", flush=True); t = time.time()
    recs = anomalies_from_map(amap, 0.5, spec.min_area_px, px, merge_px=6, solid=True)
    print(f"blobs {len(recs)} {time.time() - t:.0f}s", flush=True); t = time.time()
    res = check_candidates(src, run, spec, [(d, "speck") for d in recs])
    kept = [d for d, r in res if not r]
    print(f"checks kept {len(kept)} {time.time() - t:.0f}s", flush=True); t = time.time()
    for d in kept:
        d["group"] = "speck"
    cls = zarr.open_group(str(run / "layout.zarr"), mode="r")["0"]
    assess(kept, cls, px, source=src)
    print(f"assess {time.time() - t:.0f}s", flush=True)
