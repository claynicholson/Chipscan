"""GDS registration + CPW metrology for a wafer run, starting from another registered scan of the same
design on the same microscope (its affine is the first guess; field and frame offsets are re-measured).

    uv run python scripts/dev/register_wafer.py runs/<new wafer run> runs/<registered wafer run>
"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401
import numpy as np

from chipscan.design.wafer import WaferDesign
from chipscan.design.wafer_register import WaferRegistration, frame_offsets, measure_shift, refine_field
from chipscan.io.readers import PyramidSource
from chipscan.stages.flow import scan_mask

GDS = "data/designs/NFT1_26_1_wafer(1) - Copy.gds"
PTS = [(31000, 47000), (59500, 60500), (84000, 41000), (41000, 84500), (71000, 76000), (52000, 30500), (66000, 90000), (45000, 55000), (75000, 50000)]


def log(*a):
    print(*a, flush=True)


def check(src, wd, reg, level=0):
    return [measure_shift(src, wd, reg, c, r, level=level) for c, r in PTS]


if __name__ == "__main__":
    run, ref = Path(sys.argv[1]), Path(sys.argv[2])
    only_coarse = "--coarse" in sys.argv
    src = PyramidSource(run / "corrected.zarr")
    wd = WaferDesign.load(GDS)
    base = WaferRegistration.load(ref / "design_registration.npz")
    reg = WaferRegistration(base.affine.copy(), np.zeros((2, 2, 2)), 10**9, 0.0)
    t = time.time()
    # coarse: where does this wafer sit relative to the reference placement (big capture range first)
    for level in (4, 3, 2, 1):
        m = [x for x in check(src, wd, reg, level) if x is not None]
        log(f"level {level}: {len(m)}/{len(PTS)} matched", [tuple(round(v, 1) for v in x) for x in m])
        if len(m) >= 3:
            d = np.median(np.array(m)[:, :2], 0)
            reg.affine[:, 2] += d
            log(f"  shift by {d.round(1)} px")
    if only_coarse:
        sys.exit()
    mask = scan_mask(run, src, base.pixel_size_um)
    for level in (1, 0):
        reg, n = refine_field(src, wd, reg, step=2048, level=level, mask=mask)
        log(f"field level {level}: {n} tiles, |max| {np.abs(reg.field).max():.1f} px, {time.time() - t:.0f}s")
    reg.save(run / "design_registration.base.npz")
    period = tuple(json.loads((run / "corrected.straighten.json").read_text())["tiles"]["period_px"])
    off, score = frame_offsets(str(run / "corrected.zarr"), GDS, str(run / "design_registration.base.npz"), src.shape, period, mask=mask)
    reg.frames, reg.frame_period = off, period
    reg.save(run / "design_registration.npz")
    log(f"frames {off.shape}, measured {int((score > 0).sum())}, {time.time() - t:.0f}s")
    res = [x for x in check(src, wd, reg) if x is not None]
    log("residual at check points (dcol, drow, ncc):", [tuple(round(v, 1) for v in x) for x in res])
    json.dump({"pixel_size_um": reg.pixel_size_um, "from": str(ref), "affine_um_to_px": reg.affine.tolist()}, open(run / "design_registration.json", "w"), indent=2)

    from chipscan.stages.cpw import summarize, survey_wafer

    rows = survey_wafer(src, wd, reg, progress=lambda i, n: log(f"  cpw {i}/{n} {time.time() - t:.0f}s"))
    for r in rows:
        r["px_um"] = reg.pixel_size_um
    json.dump({"rows": rows, "summary": summarize(rows)}, open(run / "cpw.json", "w"))
    log(f"cpw: {len(rows)} sites, {time.time() - t:.0f}s")
