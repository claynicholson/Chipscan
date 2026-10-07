"""Tiling of huge images with context margins.

Each :class:`Tile` has a *core* box (what we are responsible for) and a *read*
box (core plus a margin of context on every side).  Tile cores partition the
image exactly, so a per-pixel result written from cores has no seams and
no overlap bookkeeping, while every prediction still saw `margin_px` of context.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from chipscan.geometry import BBox


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    col: int
    core: BBox  # level-0 px, half-open
    read: BBox  # level-0 px, core padded by margin (may exceed image bounds; reads are zero-padded)

    @property
    def core_slice_in_read(self) -> tuple[slice, slice]:
        """Slices that cut the core out of an array read over `read`."""
        oy = int(self.core.y0 - self.read.y0)
        ox = int(self.core.x0 - self.read.x0)
        return (slice(oy, oy + int(self.core.height)), slice(ox, ox + int(self.core.width)))

    @property
    def read_shape(self) -> tuple[int, int]:
        return (int(self.read.height), int(self.read.width))


def make_tiles(shape: tuple[int, int], tile_px: int, margin_px: int = 0, bbox: BBox | None = None) -> list[Tile]:
    """Partition `bbox` (default: whole image) into cores of at most `tile_px`."""
    h, w = shape
    b = bbox if bbox is not None else BBox(0, 0, w, h)
    x0, y0, x1, y1 = b.to_int()
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    tiles: list[Tile] = []
    i = 0
    for r, ty in enumerate(range(y0, y1, tile_px)):
        for c, tx in enumerate(range(x0, x1, tile_px)):
            core = BBox(tx, ty, min(tx + tile_px, x1), min(ty + tile_px, y1))
            read = core.pad(margin_px)
            tiles.append(Tile(i, r, c, core, read))
            i += 1
    return tiles


def grid_sample_boxes(bbox: BBox, n: int, size_px: int) -> list[BBox]:
    """n x n boxes of `size_px` spread over `bbox` (for fine registration sampling)."""
    xs = np.linspace(bbox.x0, bbox.x1 - size_px, n) if bbox.width > size_px else [bbox.x0]
    ys = np.linspace(bbox.y0, bbox.y1 - size_px, n) if bbox.height > size_px else [bbox.y0]
    out = []
    for y in ys:
        for x in xs:
            out.append(BBox(float(x), float(y), float(x) + size_px, float(y) + size_px))
    return out


def iter_batches(tiles: list[Tile], batch: int) -> Iterator[list[Tile]]:
    for i in range(0, len(tiles), batch):
        yield tiles[i : i + batch]
