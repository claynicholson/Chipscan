"""Command line interface.

    chipscan convert  IMAGE [--out X.zarr] [--pixel-size 0.92]
    chipscan layers   DESIGN.gds
    chipscan register CONFIG.yaml
    chipscan inspect  CONFIG.yaml
    chipscan train    CONFIG.yaml --out model.pt [--epochs 10] [--prior]
    chipscan synth    OUT_DIR  [--rotation 3.7 --scale 1.02 --flip]
    chipscan score    outputs/x.defects.json truth.json
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
def train(
    config: Path,
    out: Path = typer.Option(Path("models/unet.pt")),
    epochs: int = 10,
    samples: int = 2000,
    tile: int = 256,
    batch: int = 8,
    lr: float = 1e-3,
    prior: bool = typer.Option(False, help="Feed the rendered design as a second input channel"),
    device: str = "auto",
):
    """Train a UNet using the registered design as labels (no manual annotation)."""
    from chipscan.config import PipelineConfig
    from chipscan.pipeline import load_inputs
    from chipscan.pipeline import register as _register
    from chipscan.segmentation.dataset import DesignLabeledTiles
    from chipscan.segmentation.train import split_dataset
    from chipscan.segmentation.train import train as _train

    cfg = PipelineConfig.load(config)
    src, des = load_inputs(cfg)
    warp, px, _ = _register(cfg, src, des)
    ds = DesignLabeledTiles(src, des, warp, tile, samples, cfg.detection.tolerance_px, prior)
    tr, va = split_dataset(ds)
    _train(tr, out, epochs, batch, lr, cfg.segmentation.base_channels, cfg.segmentation.depth, device, va)
    console.print(f"saved {out}")


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
