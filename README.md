# chipscan

Design-referenced optical defect inspection for chips. Give it a giant stitched
microscope image and the GDS it was fabricated from; it registers the two
(rotation, scale, mirror, translation, plus a residual warp for stitching
seams), segments metal, diffs against the design, classifies the differences
into physically meaningful defect classes, measures critical dimensions, and
exports everything into the tools a fab lab already uses (KLayout markers, a
defect GDS layer, a PRECiV position list for high-magnification revisits).

Nothing in the core knows about a specific microscope, product, or layer stack:
a new chip is a new YAML file.

```
image (PNG/JPG/TIFF/JP2, any size) ──► zarr pyramid ──┐
                                                       ├─► register ─► segment tiles ─► compare ─► defects ─► export
GDS (gdstk) ──► polygons in um ──► render on demand ──┘                      └─► metrology (line cuts)
```

## Install

```bash
brew install uv vips          # macOS; uv manages Python, libvips streams giant images
uv sync --extra dev
uv run pytest -q
```

## Try it on synthetic data

```bash
uv run chipscan synth data/synthetic --rotation 3.7 --scale 1.02   # tilted, noisy chip + injected defects
uv run chipscan inspect data/synthetic/config.yaml
uv run chipscan score data/synthetic/outputs/synthetic.defects.json data/synthetic/truth.json
open data/synthetic/outputs/synthetic.overlay.png
```

On the default synthetic chip (3.7 deg tilt, 2 % scale error, mosaic brightness steps,
scratches, noise) the classical pipeline recovers the rotation to 0.02 deg, finds 40/40
injected defects with the right class and no false alarms, and measures the CPW
gap/centre/gap to within 0.15 um.  Edge-attached defects must protrude more than
`tolerance_px + 2` pixels to be visible to the tolerance-band comparison.

## Use it on real data

1. Build a pyramid once (the 1.4 GB mosaics become random-access):
   `uv run chipscan convert "/Volumes/T9/Wafer Imaging/.../B1-reticle.png" --pixel-size 0.93`
2. See what the GDS contains: `uv run chipscan layers design.gds`
3. Copy `configs/example_wafer.yaml`, set the paths, layers and polarity.
4. `uv run chipscan register my.yaml` and check the printed rotation/scale/score.
5. `uv run chipscan inspect my.yaml`
6. Load `outputs/<name>.lyrdb` in KLayout (Tools > Marker Browser) over the GDS.

## Layout

| module | role |
| --- | --- |
| `chipscan/geometry.py` | `BBox`, `Affine2D` (world um -> px), `WarpField` (affine + residual grid) |
| `chipscan/io/` | zarr pyramid builder, `ImageSource` readers, PRECiV PositionList XML |
| `chipscan/design/` | GDS loading, layer/polarity handling, on-demand rasterization |
| `chipscan/registration/` | coarse (Fourier-Mellin + phase correlation, scored hypotheses) and fine (tile grid, robust affine refit, residual field) |
| `chipscan/tiling/` | tiles with context margin; cores partition the image exactly, so no seams |
| `chipscan/segmentation/` | `Segmenter` protocol; classical baselines; UNet + self-labelled training |
| `chipscan/detection/` | tolerance-band diff, connected components, rule taxonomy, merge, line-cut metrology |
| `chipscan/export/` | KLayout `.lyrdb`, defect GDS, JSON/CSV, overlay PNG |
| `chipscan/pipeline.py` | orchestration; every stage is callable on its own |
| `chipscan/synth.py` | synthetic chip generator with ground truth |
| `chipscan/evaluate.py` | recall / class accuracy / false alarms against a truth list |

## Extending

* **New image format**: implement `ImageSource` (see `io/readers.py`). Olympus `.vsi`
  can be read with Bio-Formats (`uv sync --extra vsi`) and converted with `build_pyramid`.
* **New segmenter**: implement `Segmenter.predict(tiles, priors)`; register it in
  `segmentation/__init__.py:build_segmenter`.
* **Product-specific defect classes** (airbridge collapse, junction defects):
  implement `Classifier.classify(kind, ctx)` and pass it to `run_inspection`.
* **Training the UNet** uses the registered design as labels; no annotation needed:
  `uv run chipscan train my.yaml --out models/unet.pt --epochs 10 [--prior]`.

## Defect taxonomy (default rules)

| kind | class | rule |
| --- | --- | --- |
| extra metal | `short` | touches two or more separate design metal regions |
| extra metal | `protrusion` | attached to one design metal region |
| extra metal | `particle` | touches none |
| missing metal | `open` | removing it disconnects its design region |
| missing metal | `notch` | on the edge of a design region |
| missing metal | `void` | fully interior (pinhole) |
| either | `*_feature` | larger than `large_um2` |

All coordinates in exports are design microns, so results line up with the GDS
regardless of how the chip was placed under the microscope.
