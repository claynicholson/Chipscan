"""Build a zarr pyramid from any giant image, and open sources uniformly.

libvips streams the decode, so a 50k x 50k PNG never has to fit in RAM as float.
Each level is a chunked uint8 array; level k+1 is a 2x2 block mean of level k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from rich.progress import Progress

from chipscan.io.readers import ArraySource, ImageSource, PyramidSource, downsample2

_CHANNEL_OPS = {"gray", "r", "g", "b", "max", "min"}


def _import_pyvips():
    """Import pyvips, loading torch first.

    libvips and torch each bundle an OpenMP runtime; on macOS, loading libvips first makes
    torch's convolutions segfault later in the same process.  Loading torch first is safe.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    import pyvips

    return pyvips


def _to_single_channel(img, channel: str):
    """pyvips image -> single band pyvips image."""

    if img.bands == 1:
        return img
    if img.bands == 4:
        img = img[:3]
    if channel == "gray":
        return img.colourspace("b-w") if img.interpretation != "b-w" else img
    if channel in ("r", "g", "b"):
        return img["rgb".index(channel)]
    if channel == "max":
        return img.bandrank(index=img.bands - 1)
    if channel == "min":
        return img.bandrank(index=0)
    raise ValueError(f"unknown channel op {channel!r}; expected one of {_CHANNEL_OPS}")


def build_pyramid(
    image_path: str | Path,
    out_path: str | Path,
    pixel_size_um: float | None = None,
    channel: str = "gray",
    chunk_px: int = 1024,
    invert: bool = False,
    min_level_px: int = 512,
    overwrite: bool = False,
) -> PyramidSource:
    """Decode `image_path` with libvips and write a zarr pyramid to `out_path`."""
    pyvips = _import_pyvips()

    out_path = Path(out_path)
    st = Path(image_path).stat()
    fingerprint = {"size": st.st_size, "mtime": st.st_mtime, "channel": channel, "invert": invert}
    if out_path.exists() and not overwrite:
        existing = PyramidSource(out_path)
        if existing.attrs.get("fingerprint") == fingerprint:
            return existing
        # the image changed since the pyramid was built: rebuild rather than serve stale pixels

    img = pyvips.Image.new_from_file(str(image_path))
    img = _to_single_channel(img, channel)
    if img.format != "uchar":
        img = img.cast("uchar")
    if invert:
        img = img.invert()
    h, w = img.height, img.width

    root = zarr.open_group(str(out_path), mode="w")
    root.attrs.update(
        {
            "source": str(Path(image_path).resolve()),
            "pixel_size_um": pixel_size_um,
            "channel": channel,
            "fingerprint": fingerprint,
            "downsample": 2,
            "chunk_px": chunk_px,
        }
    )

    lvl0 = root.create_array("0", shape=(h, w), chunks=(chunk_px, chunk_px), dtype="uint8")
    n_chunks = (-(-h // chunk_px)) * (-(-w // chunk_px))
    with Progress() as prog:
        task = prog.add_task(f"level 0 ({w}x{h})", total=n_chunks)
        for y in range(0, h, chunk_px):
            for x in range(0, w, chunk_px):
                cw, ch = min(chunk_px, w - x), min(chunk_px, h - y)
                tile = img.crop(x, y, cw, ch).numpy()
                if tile.ndim == 3:
                    tile = tile[:, :, 0]
                lvl0[y : y + ch, x : x + cw] = tile
                prog.advance(task)

    _build_levels(root, chunk_px, min_level_px)
    root.attrs["levels"] = len([k for k in root.array_keys()])
    return PyramidSource(out_path)


def _build_levels(root: zarr.Group, chunk_px: int, min_level_px: int) -> None:
    level = 0
    while True:
        prev = root[str(level)]
        h, w = prev.shape
        if min(h, w) // 2 < min_level_px:
            break
        h2, w2 = h // 2, w // 2
        nxt = root.create_array(str(level + 1), shape=(h2, w2), chunks=(chunk_px, chunk_px), dtype="uint8")
        step = chunk_px * 2
        with Progress() as prog:
            task = prog.add_task(f"level {level + 1} ({w2}x{h2})", total=(-(-h2 // chunk_px)) * (-(-w2 // chunk_px)))
            for y in range(0, h2 * 2, step):
                for x in range(0, w2 * 2, step):
                    blk = np.asarray(prev[y : min(y + step, h2 * 2), x : min(x + step, w2 * 2)])
                    nxt[y // 2 : y // 2 + blk.shape[0] // 2, x // 2 : x // 2 + blk.shape[1] // 2] = downsample2(blk)
                    prog.advance(task)
        level += 1


def open_source(
    path: str | Path,
    pixel_size_um: float | None = None,
    pyramid_path: str | Path | None = None,
    channel: str = "gray",
    chunk_px: int = 1024,
    invert: bool = False,
) -> ImageSource:
    """Open an image path as an ImageSource, building a pyramid next to it if needed."""
    p = Path(path)
    if p.suffix == ".zarr" or (p.is_dir() and (p / "zarr.json").exists()):
        src = PyramidSource(p)
        if pixel_size_um is not None:
            src.pixel_size_um = pixel_size_um
        return src
    if p.suffix.lower() == ".npy":
        return ArraySource(np.load(p), pixel_size_um)
    out = Path(pyramid_path) if pyramid_path else p.with_suffix(p.suffix + ".zarr")
    src = build_pyramid(p, out, pixel_size_um, channel=channel, chunk_px=chunk_px, invert=invert)
    if pixel_size_um is not None:
        src.pixel_size_um = pixel_size_um
    return src
