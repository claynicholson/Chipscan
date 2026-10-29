"""Per camera frame GDS registration for a wafer run: uv run python scripts/dev/wafer_frames.py runs/<wafer run>"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401

from chipscan.design.wafer import WaferDesign
from chipscan.design.wafer_register import WaferRegistration, frame_offsets, measure_shift
from chipscan.io.readers import PyramidSource
from chipscan.stages.flow import scan_mask

GDS = "data/designs/NFT1_26_1_wafer(1) - Copy.gds"

if __name__ == "__main__":
    run = Path(sys.argv[1])
    src = PyramidSource(run / "corrected.zarr")
    period = tuple(json.loads((run / "corrected.straighten.json").read_text())["tiles"]["period_px"])
    reg = WaferRegistration.load(run / "design_registration.npz")
    reg.frames, reg.frame_period = None, None
    tmp = run / "design_registration.base.npz"
    reg.save(tmp)
    t = time.time()
    off, score = frame_offsets(str(run / "corrected.zarr"), GDS, str(tmp), src.shape, period, mask=scan_mask(run, src, 1.6285), progress=lambda i, n: print(f"  {i}/{n} {time.time() - t:.0f}s", flush=True))
    reg.frames, reg.frame_period = off, period
    reg.save(run / "design_registration.npz")
    print("frames", off.shape, "measured", int((score > 0).sum()), f"{time.time() - t:.0f}s", flush=True)
    wd = WaferDesign.load(GDS)
    pts = [(31000, 47000), (59500, 60500), (84000, 41000), (41000, 84500), (71000, 76000), (52000, 30500), (66000, 90000), (45000, 55000), (75000, 50000)]
    res = []
    for c, r in pts:
        m = measure_shift(src, wd, reg, c, r, level=0)
        res.append(None if m is None else (round(m[0], 1), round(m[1], 1), round(m[2], 2)))
    print("residual at check points (dcol, drow px, ncc):", res)
