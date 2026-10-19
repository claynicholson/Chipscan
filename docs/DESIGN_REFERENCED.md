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

## Design-free anomaly detection (no GDS)

```bash
uv run chipscan train-anomaly configs/train_anomaly.yaml        # scenes are just images
uv run chipscan detect data/pyramids/AFT5_07_C1.zarr --model models/anomaly_v3.pt --pixel-size 0.814
uv run chipscan serve outputs/anomaly/AFT5_07_C1.anomalies.json  # view the outlines
```

When no layout is available (or it is the wrong revision), a second detector works from
the image alone. It is trained self-supervised: tiles from any number of scenes get
synthetic anomalies injected (blobs, short rough bars, scratches, discoloration, blur
spots, erasures, all forced to contrast with their surroundings) under heavy appearance
randomisation (gamma, contrast, polarity, blur, noise, 0.6x-1.6x scale, mosaic steps), and
a UNet learns to outline them. Trained on two chip designs, a PCB panel and synthetic
chips, validated on a reticle it never saw: IoU 0.87, recall 0.94, precision 0.92 on
injected anomalies.

Labels and markings (letters, digits, unit strings) are designed structure too, so training
tiles also receive random *unlabelled* synthetic text in many fonts, sizes and orientations;
without it the chip labels were the dominant false alarms on real reticles.

A learned detector alone cannot tell unfamiliar *designed* structure from a defect, so
`detect` adds **repeat suppression**: it estimates the image's repetition pitches by
autocorrelation (it finds the chip pitch on a reticle by itself) and drops every
candidate that is reproduced one pitch away. Structure repeats; defects do not. On a real
AFT5 reticle this halves the candidates and leaves contamination (a fiber, a particle)
plus the per-chip labels and split-to-split design differences, which a design-free
method cannot resolve.

## Web viewer

```bash
uv run chipscan serve data/synthetic/config.yaml other_run.yaml --port 8000   # add --host 0.0.0.0 for the lab network
```

A local web app (FastAPI + OpenSeadragon) over any number of inspected runs:

* deep-zoom of the full mosaic, tiles cut on demand from the zarr pyramid and cached;
* overlays: design edges rendered per tile through the saved registration, the model's
  probability heatmap, and defect outlines coloured by class;
* defect list with thumbnails, filtering (`short`, `>50`, `unreviewed`, `#12`), sorting, and
  keyboard review: `r` real, `f` false alarm, `u` unsure, `1`-`6` reclassify, `x` select;
* reviews go to `reviews.sqlite` next to the outputs and come back out as CSV, so hand
  labelling is a by-product of looking at results;
* `m` measures metal/gap widths along a drawn line; "re-image selected" downloads a PRECiV
  PositionList of the selected defects for the microscope to revisit at high magnification.

## Layout

| module | role |
| --- | --- |
| `chipscan/geometry.py` | `BBox`, `Affine2D` (world um -> px), `WarpField` (affine + residual grid) |
| `chipscan/io/` | zarr pyramid builder, `ImageSource` readers, PRECiV PositionList XML |
| `chipscan/design/` | GDS loading, layer/polarity handling, on-demand rasterization |
| `chipscan/registration/` | coarse (Fourier-Mellin + phase correlation, scored hypotheses) and fine (tile grid, robust affine refit, residual field) |
| `chipscan/tiling/` | tiles with context margin; cores partition the image exactly, so no seams |
| `chipscan/segmentation/` | `Segmenter` protocol; classical baselines; 3-head UNet, injection dataset, trainer |
| `chipscan/detection/` | tolerance-band diff, connected components, rule taxonomy, merge, line-cut metrology |
| `chipscan/export/` | KLayout `.lyrdb`, defect GDS, JSON/CSV, overlay PNG |
| `chipscan/pipeline.py` | orchestration; every stage is callable on its own |
| `chipscan/web/` | FastAPI tile/defect/review server and the OpenSeadragon front end |
| `chipscan/synth.py` | synthetic chip generator with ground truth |
| `chipscan/evaluate.py` | recall / class accuracy / false alarms against a truth list |

## Extending

* **New image format**: implement `ImageSource` (see `io/readers.py`). Olympus `.vsi`
  can be read with Bio-Formats (`uv sync --extra vsi`) and converted with `build_pyramid`.
* **New segmenter**: implement `Segmenter.predict(tiles, priors)`; register it in
  `segmentation/__init__.py:build_segmenter`.
* **Product-specific defect classes** (airbridge collapse, junction defects):
  implement `Classifier.classify(kind, ctx)` and pass it to `run_inspection`.
* **Training the UNet** needs no annotation: `uv run chipscan train configs/train_example.yaml`.
  See "Training a detector" below.

## Training a detector

The classical baseline thresholds the image and diffs it against the design outside a
tolerance band. The learned detector is trained to output defect maps directly, with
no hand labels, from two free signals:

1. **The design is the label.** Tiles are drawn from registered scenes; the rendered
   design is the metal target (with a 1 px ignore band) and, optionally, an input
   channel. The input prior is jittered by a pixel so registration slop is learned as
   normal.
2. **Design-aware defect injection.** Guided by the design, metal-textured blobs are
   pasted into gaps (shorts, particles, protrusions) and substrate-textured blobs onto
   metal (opens, voids, notches); textures come from the tile itself. The injected
   regions are the `extra` / `missing` targets.

Feeding the design as an input while using it as the only label is degenerate (the net
copies the prior); the injected defects are what force it to look at the image.

```bash
uv run chipscan train configs/train_example.yaml                     # scenes listed in the YAML
uv run python scripts/benchmark_synthetic.py --model models/unet_synthetic.pt   # classical vs UNet
```

On three held-out synthetic chips (different seeds, tilts of 2 to 92 degrees, scale
errors, one mirrored; 90 injected defects), after 12 epochs on one synthetic scene:

| method | recall | class accuracy | false alarms |
| --- | --- | --- | --- |
| classical (design-prior threshold, 2 px tolerance band) | 82/90 (91%) | 81/90 (90%) | 1 |
| UNet detector (injection-trained, no hand labels) | 90/90 (100%) | 88/90 (98%) | 0 |

The classical misses are protrusions and voids below the tolerance-band floor; the
detector has no such floor. Its two "errors" are 10 um protrusions that physically
reach the centre conductor, which it correctly calls shorts.

Then set `segmentation.method: unet` and `model_path` in a pipeline YAML. A 3-output
model is used as a detector (its defect channels replace the tolerance-band diff); a
1-output model is used as a plain segmenter. Add every registered reticle you have as a
scene; appearance variety is what makes the model robust on real mosaics.

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
