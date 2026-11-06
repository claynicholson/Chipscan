<img align="center" width="1296" height="360" alt="Group 94 (3)" src="https://github.com/user-attachments/assets/149f89e9-35f7-4fb3-bb2c-ca95c2ab493b" />

<p align="center">
  Automated optical inspection for wafers.<br>
  Microscope scan in -> annotated defects out
</p>

---

## What it does

A wafer-scale brightfield scan (such as the Olympus DSX1000, ~0.8 µm/px, billions of pixels) is far too slow to inspect
by eye. Chip Scanner reads it, straightens and fixes the stitching in the scan, then finds and ranks defects in three groups:

| Group | What it finds | How |
|---|---|---|
| **Specks** | dust, particles, fibres, residue, probe scrub marks | dedicated speck UNet + reference residual against identical chips + pad rule |
| **Trace errors** | opens, shorts, bridges, nicks on wiring | dedicated trace UNet (off by default while being validated) |
| **Component errors** | Josephson junctions that differ from their neighbours | look-alike junctions grouped into lines, each compared with the median of its line |

Each finding gets a **risk score from 0 to 100**. It goes up with the defect's size and with how close it sits to a
junction or a trace. Dust on bare ground plane ranks low; a particle touching a JJ ranks critical.

## Pipeline

```
scan ─► straighten ─► layout ─► specks ─► traces ─► components ─► verify ─► risk ─► viewer
        tilt + stitch   where are   UNet +     UNet      line-wise       compare      distance
        frame shading   traces/JJs  reference            outliers        with twins   to JJs/traces
```

1. **Straighten.** Measures the stage tilt and the camera-frame grid of the mosaic, then removes per-frame shading,
   so stitching seams do not turn into "defects".
2. **Layout.** A 4-class network (background, trace, component, text) maps the chip when no GDS is given. Lettering
   and test structures are recognised, so labels are never reported as dirt.
3. **Detect.** Specks are found three independent ways and merged:
   - a speck UNet, trained self-supervised by painting realistic dirt onto real scans;
   - a **reference residual**: every 384 px tile is aligned with its identical twins elsewhere on the scan, and
     anything only this copy has is dirt;
   - a rule for marks on bond pads.
4. **Verify.** Every candidate is re-checked against the same spot on its twins (1 px tolerance, median over
   twins). Designed structure is explained away. A dedicated test catches stitching ghosts, where two misregistered
   camera frames are blended.
5. **Rank and review.** Findings are risk-scored and served in a web viewer with deep zoom, group and risk filters,
   live sensitivity, and a review queue.

No design file is required. With a GDS, the layout comes from the design instead of the network.

## Installation Guide

```bash
brew install uv vips
uv sync
uv run chipscan add /path/to/scan.png --name AFT5_C1 --pixel-size 0.814   # full scan, any size
uv run chipscan serve --runs runs                                         # http://localhost:8000
```

## Measuring it

`scripts/eval_specks.py` plants realistic specks into a held-out scan that no model was trained on. It runs the
complete pipeline, filters and twin checks included, and reports recall by speck size and contrast. It also reports
the alarm rate on the same windows without planted specks.


Design-referenced inspection against a GDS (registration, CD metrology, KLayout export) is documented in
[docs/DESIGN_REFERENCED.md](docs/DESIGN_REFERENCED.md).

<p align="center"><sub>Research Laboratory of Electronics · Massachusetts Institute of Technology</sub></p>
