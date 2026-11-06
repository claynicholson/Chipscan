"""Run the component check at a low z on a run and draw every flag (sorted by z) into a contact sheet.

    uv run python scripts/dev/show_components.py runs/<run> out.jpg [z]
"""
import json
import sys
from pathlib import Path

import torch  # noqa: F401
import cv2
import numpy as np
import zarr

from chipscan.io.readers import PyramidSource
from chipscan.stages.components import component_errors, find_instances

if __name__ == "__main__":
    run, out = Path(sys.argv[1]), sys.argv[2]
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    meta = json.loads((run / "scan.json").read_text())
    src = PyramidSource(run / "corrected.zarr")
    cls = zarr.open_group(str(run / "layout.zarr"), mode="r")["0"]
    recs, summ = component_errors(src, find_instances(cls), meta["spec"]["pixel_size_um"], z_min=z, class_map=cls, report_missing=False)
    print(json.dumps(summ), len(recs), "flags", flush=True)
    recs.sort(key=lambda d: -d["z"])
    json.dump([{k: d[k] for k in ("z", "centroid_px", "bbox_px")} for d in recs], open(out + ".json", "w"))
    tiles = []
    for d in recs[:48]:
        b = d["component_bbox_px"]
        cx, cy = int((b["x0"] + b["x1"]) / 2), int((b["y0"] + b["y1"]) / 2)
        R = 45
        a = cv2.cvtColor(src.read_region(cx - R, cy - R, cx + R, cy + R), cv2.COLOR_GRAY2BGR)
        for q in d["polygons_px"]:
            cv2.polylines(a, [np.rint(np.asarray(q) - [cx - R, cy - R]).astype(np.int32)], True, (0, 0, 255), 1)
        a = cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(a, f"z{d['z']:.0f}", (3, 14), 0, 0.45, (255, 0, 0), 1)
        tiles.append(a)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]))
