"""Command line interface.

    chipscan convert  IMAGE [--out X.zarr] [--pixel-size 0.92]
    chipscan layers   DESIGN.gds
    chipscan register CONFIG.yaml
    chipscan inspect  CONFIG.yaml
    chipscan train    TRAIN.yaml
    chipscan synth    OUT_DIR  [--rotation 3.7 --scale 1.02 --flip]
    chipscan score    outputs/x.defects.json truth.json
    chipscan train-anomaly TRAIN.yaml        (design-free)
    chipscan detect   IMAGE --model anomaly.pt (design-free)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Design-referenced optical chip inspection.", no_args_is_help=True)
console = Console()


@app.command()
def convert(
    image: Path,
    out: Path | None = typer.Option(None, help="Output .zarr (default: next to image)"),
    pixel_size: float | None = typer.Option(None, help="Microns per pixel"),
    channel: str = typer.Option("gray", help="gray|r|g|b|max|min"),
    chunk: int = 1024,
    invert: bool = False,
    overwrite: bool = False,
):
    """Build a zarr pyramid from a giant PNG/JPG/TIFF/JP2."""
    from chipscan.io.pyramid import build_pyramid

    out = out or image.with_suffix(image.suffix + ".zarr")
    src = build_pyramid(image, out, pixel_size, channel, chunk, invert, overwrite=overwrite)
    console.print(f"wrote {out}: {src.shape[1]}x{src.shape[0]} px, {src.num_levels} levels")


@app.command()
def layers(design: Path):
    """List GDS layers and polygon counts."""
    from chipscan.design import list_layers

    t = Table("layer/datatype", "polygons")
    for (layer, dt), n in sorted(list_layers(design).items()):
        t.add_row(f"{layer}/{dt}", str(n))
    console.print(t)


@app.command()
def register(config: Path, out: Path | None = None):
    """Solve rotation/scale/flip/translation + local warp; save registration JSON."""
    from chipscan.config import PipelineConfig
    from chipscan.pipeline import load_inputs
    from chipscan.pipeline import register as _register

    cfg = PipelineConfig.load(config)
    src, des = load_inputs(cfg)
    warp, px, info = _register(cfg, src, des)
    out = out or cfg.export.out_dir / f"{cfg.name}.registration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    warp.save(out)
    console.print(f"pixel size {px:.5f} um/px; saved {out}")


@app.command()
def inspect(config: Path):
    """Run the full pipeline and write all exports."""
    from chipscan.config import PipelineConfig
    from chipscan.pipeline import run_inspection

    cfg = PipelineConfig.load(config)
    res = run_inspection(cfg)
    t = Table("class", "count")
    from collections import Counter

    for k, v in sorted(Counter(d.cls for d in res.defects).items()):
        t.add_row(k, str(v))
    console.print(t)
    for c in res.cuts:
        console.print(f"cut {c.name}: " + ", ".join(f"{p}={length:.2f}um" for p, length in c.runs))


@app.command()
def train(config: Path):
    """Train the inspection UNet from a training YAML (scenes + hyper-parameters). No hand labels needed."""
    from chipscan.config import PipelineConfig, TrainConfig
    from chipscan.pipeline import load_scene
    from chipscan.segmentation.dataset import InjectionConfig, InspectionTiles, split_scene
    from chipscan.segmentation.train import train as _train

    tc = TrainConfig.load(config)
    train_scenes, val_scenes = [], []
    for sp in tc.scenes:
        sc = load_scene(PipelineConfig.load(sp))
        tr, va = split_scene(sc, tc.val_frac)
        train_scenes.append(tr)
        val_scenes.append(va)
    inj = InjectionConfig(**tc.injection.model_dump())
    ds = InspectionTiles(train_scenes, tc.tile_px, tc.samples_per_epoch, tc.ignore_px, tc.use_prior, tc.prior_jitter_px, True, inj, tc.seed)
    vs = InspectionTiles(val_scenes, tc.tile_px, max(tc.samples_per_epoch // 10, 32), tc.ignore_px, tc.use_prior, 0.0, False, inj, tc.seed + 1)
    _train(ds, tc.out, tc.epochs, tc.batch_size, tc.lr, tc.base_channels, tc.depth, tc.device, vs, tc.num_workers, tc.defect_pos_weight, tc.init_from)
    console.print(f"saved {tc.out} (+ .history.json)")


@app.command(name="train-anomaly")
def train_anomaly_cmd(config: Path):
    """Train the design-free anomaly detector (image in, anomaly outline out) from a training YAML."""
    from chipscan.config import AnomalyTrainConfig
    from chipscan.geometry import BBox
    from chipscan.io import open_source
    from chipscan.segmentation.anomaly import AnomalyInjection, AnomalyTiles
    from chipscan.segmentation.anomaly_train import train_anomaly

    tc = AnomalyTrainConfig.load(config)

    def scenes(lst):
        srcs, regs = [], []
        for sc in lst:
            srcs.append(open_source(sc.path, sc.pixel_size_um, None, sc.channel))
            regs.append(BBox(*sc.region_px) if sc.region_px else None)
        return srcs, regs

    tr_src, tr_reg = scenes(tc.scenes)
    inj = AnomalyInjection(**tc.injection)
    ds = AnomalyTiles(tr_src, tc.tile_px, tc.samples_per_epoch, inj, True, tc.seed, regions=tr_reg)
    vs = None
    if tc.val_scenes:
        va_src, va_reg = scenes(tc.val_scenes)
        vs = AnomalyTiles(va_src, tc.tile_px, tc.val_samples, inj, False, tc.seed + 1, regions=va_reg)
    train_anomaly(ds, tc.out, tc.epochs, tc.batch_size, tc.lr, tc.base_channels, tc.depth, tc.device, vs, tc.num_workers, tc.pos_weight, tc.init_from)
    console.print(f"saved {tc.out}")


@app.command()
def detect(
    image: Path,
    model: Path = typer.Option(..., help="Anomaly model .pt"),
    out: Path = typer.Option(Path("outputs/anomaly"), help="Output directory"),
    name: str | None = None,
    pixel_size: float | None = typer.Option(None, help="Microns per pixel (for areas in um2)"),
    threshold: float = 0.5,
    min_area_px: int = 6,
    tile: int = 512,
    device: str = "auto",
    suppress_repeats: bool = typer.Option(True, help="Drop candidates reproduced one repetition pitch away (designed structure)"),
    match_ncc: float = typer.Option(0.8, help="NCC above which a candidate counts as repeated"),
):
    """Design-free anomaly detection on any image or pyramid: outlines, JSON/CSV, overlay. No GDS."""
    import json

    import zarr

    from chipscan.io import open_source
    from chipscan.segmentation.anomaly import (
        AnomalyDetector,
        anomalies_from_map,
        write_anomaly_outputs,
    )
    from chipscan.segmentation.repeats import estimate_pitches
    from chipscan.segmentation.repeats import suppress_repeats as _suppress

    src = open_source(image, pixel_size)
    name = name or image.stem.replace(".zarr", "")
    out.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(out / f"{name}.anomaly.zarr"), mode="w")
    store = g.create_array("0", shape=src.shape, chunks=(1024, 1024), dtype="uint8")
    det = AnomalyDetector(model, device)
    det.run(src, tile_px=tile, store=store)
    recs = anomalies_from_map(store, threshold, min_area_px, pixel_size or src.pixel_size_um)
    n_raw = len(recs)
    if suppress_repeats and recs:
        pitches = estimate_pitches(src)
        console.print(f"repetition pitches (px): x={[round(p) for p in pitches['x']]} y={[round(p) for p in pitches['y']]}")
        recs, dropped = _suppress(src, recs, pitches, match_ncc=match_ncc)
        (out / f"{name}.suppressed.json").write_text(json.dumps({"n": len(dropped), "defects": dropped}, indent=2))
        console.print(f"{n_raw} candidates -> {len(recs)} after repeat suppression ({len(dropped)} reproduced one pitch away)")
    written = write_anomaly_outputs(out, name, recs, src, store, pixel_size or src.pixel_size_um)
    console.print(f"{len(recs)} anomalies; wrote " + ", ".join(f"{k}={v.name}" for k, v in written.items()))


@app.command()
def score(
    defects_json: Path,
    truth_json: Path,
    radius_um: float = typer.Option(6.0, help="Match radius (um); the truth size is used when larger"),
    out: Path | None = typer.Option(None, help="Write the full score as JSON"),
):
    """Score an inspection's defects.json against a ground-truth JSON (synthetic or hand-labelled)."""
    import json

    from chipscan.evaluate import score_files

    sc = score_files(defects_json, truth_json, radius_um)
    console.print(str(sc))
    for m in sc.misses:
        console.print(f"  missed/misclassified: {m['class']} {m.get('size_um', 0):.1f}um at ({m['x_um']:.0f},{m['y_um']:.0f}) -> {m['near']}")
    if out:
        out.write_text(json.dumps(sc.to_dict(), indent=2))


@app.command()
def gallery(
    config: Path,
    n: int = typer.Option(24, help="Number of defects to show (most severe first)"),
    out: Path | None = None,
    truth: Path | None = typer.Option(None, help="Optional truth.json to mark injected defect centres"),
):
    """Montage of detected defects: image crop, design overlay, predicted mask, diff. Reads the outputs of `inspect`."""
    import json

    import zarr

    from chipscan.config import PipelineConfig
    from chipscan.detection.defects import Defect
    from chipscan.geometry import BBox, WarpField
    from chipscan.pipeline import load_inputs
    from chipscan.viz.gallery import defect_gallery

    cfg = PipelineConfig.load(config)
    src, des = load_inputs(cfg)
    od = cfg.export.out_dir
    warp = WarpField.load(od / f"{cfg.name}.registration.json")
    rec = json.loads((od / f"{cfg.name}.defects.json").read_text())["defects"]
    # rebuild lightweight Defect objects (polygons come from the GDS export when present)
    defs = []
    polys_by_id: dict[int, list] = {}
    gds_path = od / f"{cfg.name}.defects.gds"
    if gds_path.exists():
        import gdstk
        import numpy as np

        lib = gdstk.read_gds(str(gds_path))
        cell = lib.top_level()[0]
        labels = {(round(lab.origin[0], 3), round(lab.origin[1], 3)): int(lab.text.split("#")[-1]) for lab in cell.labels}
        for p in cell.polygons:
            # polygons are not tagged with ids in GDS; attach by nearest label centroid
            c = np.asarray(p.points).mean(axis=0)
            if labels:
                k = min(labels, key=lambda q: (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2)
                polys_by_id.setdefault(labels[k], []).append(np.asarray(p.points, dtype=float))
    for r in rec:
        bp = r.get("bbox_px")
        defs.append(
            Defect(
                r["id"], r["kind"], r["class"], BBox(**r["bbox_um"]), tuple(r["centroid_um"]), r["area_um2"], r["area_px"], r["score"],
                polys_by_id.get(r["id"], []), BBox(**bp) if bp else None, r.get("tile", -1),
            )
        )
    pred = None
    pz = od / f"{cfg.name}.pred.zarr"
    if pz.exists():
        pred = zarr.open_group(str(pz), mode="r")["0"]
    tr = json.loads(truth.read_text())["defects"] if truth else None
    out = out or od / f"{cfg.name}.gallery.png"
    defect_gallery(src, des, warp, defs, out, pred, n, truth=tr)
    console.print(f"wrote {out}")


@app.command()
def serve(
    configs: list[Path] = typer.Argument(..., help="Pipeline YAMLs whose `inspect` outputs exist, and/or <name>.anomalies.json files from `chipscan detect`"),
    host: str = typer.Option("127.0.0.1", help="Use 0.0.0.0 to serve the lab network"),
    port: int = 8000,
    db: Path | None = typer.Option(None, help="SQLite file for review decisions (default: next to the first run's outputs)"),
):
    """Web viewer: deep-zoom results with design/heatmap overlays, defect review, metrology, re-image export."""
    from chipscan.web.server import serve as _serve

    console.print(f"open http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    _serve(configs, host, port, db)


@app.command()
def synth(
    out_dir: Path,
    rotation: float = typer.Option(3.7, help="Tilt of the image relative to the design, degrees"),
    scale: float = typer.Option(1.0, help="Pixel-size error factor"),
    flip: bool = False,
    pixel_size: float = 1.0,
    size_um: float = 3000.0,
    n_defects: int = 40,
    seed: int = 0,
):
    """Generate a synthetic GDS + tilted/noisy image with injected defects, plus a config."""
    from chipscan.synth import make_synthetic

    paths = make_synthetic(out_dir, rotation, scale, flip, pixel_size, size_um, n_defects, seed)
    for k, v in paths.items():
        console.print(f"{k}: {v}")


if __name__ == "__main__":
    app()
