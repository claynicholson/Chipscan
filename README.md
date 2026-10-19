<p align="center">
  <img src="chipscan/web/static/brand/rle-at-mit.png" alt="RLE at MIT" height="96">
</p>

<h1 align="center">Chip Scanner</h1>

<p align="center">
  Automated optical inspection for superconducting quantum chips.<br>
  One stitched microscope scan in → every speck, trace error and junction defect out, ranked by how likely it is to hurt the device.
</p>

---

## What it does

A wafer-scale brightfield scan (Olympus DSX1000, ~0.8 µm/px, billions of pixels) is far too large to inspect
by eye. Chip Scanner reads it, repairs the scan itself, then finds and ranks defects in three groups:

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

## Quick start

```bash
brew install uv vips
uv sync
uv run chipscan add /path/to/scan.png --name AFT5_C1 --pixel-size 0.814   # full scan, any size
uv run chipscan serve --runs runs                                         # http://localhost:8000
```

You can also drag a scan into the viewer's **Add scan** dialog. On MIT ORCD Engaging, `scripts/engaging/run_scan.sbatch`
runs the same flow on one GPU.

## Measuring it

`scripts/eval_specks.py` plants realistic specks into a held-out scan that no model was trained on. It runs the
complete pipeline, filters and twin checks included, and reports recall by speck size and contrast. It also reports
the alarm rate on the same windows without planted specks.

## Layout

```
chipscan/stages/   straighten · nets (UNets, training) · inject (painters) · reference · verify · components · risk · flow
chipscan/web/      FastAPI + OpenSeadragon viewer
scripts/           training, evaluation, cluster jobs
```

Design-referenced inspection against a GDS (registration, CD metrology, KLayout export) is documented in
[docs/DESIGN_REFERENCED.md](docs/DESIGN_REFERENCED.md).

<p align="center"><sub>Research Laboratory of Electronics · Massachusetts Institute of Technology</sub></p>
