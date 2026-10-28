"""A full-wafer GDS without flattening it.

A wafer layout is a handful of reticle cells placed many times (NFT1: RETICLE_A/B/C x 12, plus
alignment marks and corner squares).  Flattening the top cell means ~10 M polygons; instead each
distinct child cell is rendered once at the wanted resolution and pasted at its placements.

`WaferDesign.render(px_um, bbox_um)` gives a metal raster of any window of the wafer in
image orientation (rows go down = design y goes down), cached per child cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gdstk
import numpy as np

from chipscan.design.gds import _parse_layer

_SHIFT = 4


@dataclass
class Placement:
    cell: str
    origin: tuple[float, float]
    rotation: float = 0.0
    x_reflection: bool = False


@dataclass
class WaferDesign:
    path: str
    top: str
    layers: list[tuple[int, int]]
    placements: list[Placement]
    bbox: tuple[float, float, float, float]  # um: x0, y0, x1, y1
    _polys: dict = field(default_factory=dict, repr=False)  # cell -> list of (N,2) arrays (cell coords)
    _boxes: dict = field(default_factory=dict, repr=False)  # cell -> (n, 4) polygon bboxes (cell coords)
    _lib: gdstk.Library | None = field(default=None, repr=False)

    @staticmethod
    def load(path: str | Path, top: str = "WAFER_TOP", layers: list[str] | None = None) -> WaferDesign:
        lib = gdstk.read_gds(str(path))
        t = next(c for c in lib.cells if c.name == top)
        lays = [_parse_layer(s) for s in (layers or ["5/0"])]
        pl = [Placement(r.cell.name if hasattr(r.cell, "name") else str(r.cell), tuple(map(float, r.origin)), float(r.rotation or 0.0), bool(r.x_reflection)) for r in t.references]
        (x0, y0), (x1, y1) = t.bounding_box()
        wd = WaferDesign(str(path), top, lays, pl, (float(x0), float(y0), float(x1), float(y1)))
        wd._lib = lib
        return wd

    def cell_polygons(self, name: str) -> list[np.ndarray]:
        if name not in self._polys:
            c = next(c for c in self._lib.cells if c.name == name)
            out = []
            for layer, dt in self.layers:
                for p in c.get_polygons(apply_repetitions=True, include_paths=True, depth=None, layer=layer, datatype=dt):
                    if len(p.points) >= 3:
                        out.append(np.asarray(p.points, np.float64))
            self._polys[name] = out
            self._boxes[name] = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in out]).reshape(-1, 4)
        return self._polys[name]

    def render(self, px_um: float, bbox_um: tuple[float, float, float, float] | None = None) -> np.ndarray:
        """uint8 0/1 raster of the window `bbox_um` (default: whole wafer), rows top-down
        (design y decreasing), columns left-right, `px_um` microns per pixel."""
        x0, y0, x1, y1 = bbox_um or self.bbox
        w, h = int(np.ceil((x1 - x0) / px_um)), int(np.ceil((y1 - y0) / px_um))
        canvas = np.zeros((h, w), np.uint8)
        s = 1 << _SHIFT
        for pl in self.placements:
            polys = self.cell_polygons(pl.cell)
            if not polys:
                continue
            th = np.deg2rad(pl.rotation)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            if pl.x_reflection:
                R = R @ np.diag([1.0, -1.0])
            o = np.asarray(pl.origin)
            # the window in this cell's own coordinates -> only polygons that can touch it
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) - o
            cc = corners @ R  # inverse rotation (R orthonormal)
            lo, hi = cc.min(0), cc.max(0)
            b = self._boxes[pl.cell]
            sel = np.nonzero((b[:, 2] >= lo[0]) & (b[:, 0] <= hi[0]) & (b[:, 3] >= lo[1]) & (b[:, 1] <= hi[1]))[0]
            if not len(sel):
                continue
            pts = []
            for p in (polys[i] for i in sel):
                q = p @ R.T + o
                qx = (q[:, 0] - x0) / px_um
                qy = (y1 - q[:, 1]) / px_um  # image rows go down
                pts.append(np.rint(np.stack([qx, qy], 1) * s).astype(np.int32))
            cv2.fillPoly(canvas, pts, 1, lineType=cv2.LINE_8, shift=_SHIFT)
        return canvas
