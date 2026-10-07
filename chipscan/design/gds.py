"""GDS loading and rasterization.

A :class:`Design` is a flat list of polygons in microns (the union of the chosen
layers = "metal", after applying polarity).  :meth:`Design.render` draws the
expected binary mask for any pixel box of the image given a world->px transform,
which is the primitive every other stage (registration, training labels,
comparison) is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import gdstk
import numpy as np

from chipscan.geometry import Affine2D, BBox

_SHIFT = 4  # fixed-point bits for sub-pixel fillPoly


def _parse_layer(spec: str) -> tuple[int, int] | None:
    spec = spec.strip()
    if spec in ("*", ""):
        return None
    if "/" in spec:
        a, b = spec.split("/", 1)
        return int(a), int(b)
    return int(spec), 0


@dataclass
class Design:
    polygons: list[np.ndarray]  # each (N,2) float64 in um
    bbox: BBox
    cell_name: str
    source: str = ""
    polarity: str = "drawn_is_metal"
    region: BBox | None = None  # when drawn_is_etch, the ground-plane extent

    # ---- loading ------------------------------------------------------
    @staticmethod
    def load(
        path: str | Path,
        cell: str | None = None,
        layers: list[str] | None = None,
        polarity: str = "drawn_is_metal",
        region_um: list[float] | None = None,
    ) -> Design:
        lib = gdstk.read_gds(str(path))
        tops = lib.top_level()
        if not tops:
            raise ValueError(f"{path}: no top-level cells")
        if cell is not None:
            c = next((t for t in lib.cells if t.name == cell), None)
            if c is None:
                raise ValueError(f"{path}: cell {cell!r} not found")
        else:
            c = max(tops, key=lambda t: _cell_area(t))
        specs = [_parse_layer(s) for s in (layers or ["*"])]
        wanted = None if any(s is None for s in specs) else set(specs)  # type: ignore[arg-type]
        polys: list[np.ndarray] = []
        for p in c.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            if wanted is not None and (p.layer, p.datatype) not in wanted:
                continue
            pts = np.asarray(p.points, dtype=np.float64)
            if len(pts) >= 3:
                polys.append(pts)
        if not polys:
            raise ValueError(f"{path}: no polygons on layers {layers} in cell {c.name!r}")
        d = Design.from_polygons(polys, cell_name=c.name, source=str(path), polarity=polarity)
        if region_um is not None:
            d = d.crop(BBox(*region_um))
        return d

    @staticmethod
    def from_polygons(
        polys: list[np.ndarray], cell_name: str = "TOP", source: str = "", polarity: str = "drawn_is_metal"
    ) -> Design:
        allpts = np.vstack(polys)
        bb = BBox.of_points(allpts)
        d = Design(polygons=polys, bbox=bb, cell_name=cell_name, source=source, polarity=polarity)
        d.region = bb if polarity == "drawn_is_etch" else None
        d._index()
        return d

    def _index(self) -> None:
        self._bboxes = np.array(
            [[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in self.polygons],
            dtype=np.float64,
        ).reshape(-1, 4)

    def crop(self, region: BBox) -> Design:
        """Keep polygons overlapping `region` (not clipped), and shrink the design bbox."""
        keep = [
            p for p, b in zip(self.polygons, self._bboxes, strict=True)
            if BBox(*b).overlaps(region)
        ]
        d = Design.from_polygons(keep, self.cell_name, self.source, self.polarity)
        d.bbox = region
        d.region = region if self.polarity == "drawn_is_etch" else None
        return d

    # ---- queries ------------------------------------------------------
    def polygons_in(self, world_bbox: BBox) -> list[np.ndarray]:
        b = self._bboxes
        m = (b[:, 2] >= world_bbox.x0) & (b[:, 0] <= world_bbox.x1) & (b[:, 3] >= world_bbox.y0) & (b[:, 1] <= world_bbox.y1)
        return [self.polygons[i] for i in np.nonzero(m)[0]]

    @property
    def num_polygons(self) -> int:
        return len(self.polygons)

    # ---- rendering ----------------------------------------------------
    def render(
        self,
        world_to_px: Affine2D,
        px_bbox: BBox,
        pad_um: float = 0.0,
        supersample: int | None = None,
        coverage: bool = False,
    ) -> np.ndarray:
        """Render the metal mask (uint8 0/1) for `px_bbox` of the image.

        `world_to_px` maps design um -> level-0 px.  `px_bbox` is in the *same* px
        frame; its size defines the output shape.  A pixel i covers [i, i+1); it is
        metal when the polygon covers at least half of it.  cv2.fillPoly always paints
        its outline, which biases edges outward by ~0.5 px, so we rasterize at
        `supersample`x and majority-vote (auto: 4x for tile-sized renders, less for huge ones).
        With `coverage=True` the fractional area coverage (float32 in [0,1]) is returned
        instead of a vote, which is what a physically faithful synthetic image needs.
        """
        x0, y0, x1, y1 = px_bbox.to_int()
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            return np.zeros((max(h, 0), max(w, 0)), dtype=np.float32 if coverage else np.uint8)
        if coverage and supersample is None:
            supersample = 8
        if supersample is None:
            n = h * w
            supersample = 4 if n <= 2_500_000 else (2 if n <= 12_000_000 else 1)
        ss = max(1, int(supersample))
        canvas = np.zeros((h * ss, w * ss), dtype=np.uint8)
        inv = world_to_px.inverse()
        wb = inv.apply_bbox(BBox(x0, y0, x1, y1)).pad(pad_um)
        polys = self.polygons_in(wb)
        local = Affine2D.scaling(ss).compose(Affine2D.translation(-x0, -y0).compose(world_to_px))
        scale = 1 << _SHIFT
        if polys:
            pts = [np.rint(local.apply(p) * scale).astype(np.int32) for p in polys]
            cv2.fillPoly(canvas, pts, 1, lineType=cv2.LINE_8, shift=_SHIFT)
        if self.polarity == "drawn_is_etch":
            # drawn = gaps in a ground plane: metal = region minus drawn
            region = np.zeros_like(canvas)
            if self.region is not None:
                rp = np.rint(local.apply(self.region.corners()) * scale).astype(np.int32)
                cv2.fillPoly(region, [rp], 1, lineType=cv2.LINE_8, shift=_SHIFT)
            canvas = region & (1 - canvas)
        if ss == 1:
            return canvas.astype(np.float32) if coverage else canvas
        votes = canvas.reshape(h, ss, w, ss).sum(axis=(1, 3), dtype=np.uint16)
        if coverage:
            return (votes.astype(np.float32) / float(ss * ss)).astype(np.float32)
        return (votes * 2 >= ss * ss).astype(np.uint8)

    def render_world(self, pixel_size_um: float, world_bbox: BBox | None = None) -> tuple[np.ndarray, Affine2D]:
        """Render the whole design (or a world box) at a given resolution.

        Returns (mask, world_to_px) where px is the frame of the returned mask.
        """
        wb = world_bbox or self.bbox
        s = 1.0 / pixel_size_um
        t = Affine2D.translation(-wb.x0 * s, -wb.y0 * s).compose(Affine2D.scaling(s))
        px = t.apply_bbox(wb)
        return self.render(t, BBox(0, 0, px.x1, px.y1)), t

    # ---- io -------------------------------------------------------------
    def write_gds(self, path: str | Path, layer: int = 1, datatype: int = 0) -> None:
        lib = gdstk.Library()
        c = lib.new_cell(self.cell_name)
        for p in self.polygons:
            c.add(gdstk.Polygon(p, layer=layer, datatype=datatype))
        lib.write_gds(str(path))


def _cell_area(c: gdstk.Cell) -> float:
    bb = c.bounding_box()
    if bb is None:
        return 0.0
    (x0, y0), (x1, y1) = bb
    return (x1 - x0) * (y1 - y0)


def list_layers(path: str | Path) -> dict[tuple[int, int], int]:
    """Count polygons per (layer, datatype) across the top cell. Handy for picking layers."""
    lib = gdstk.read_gds(str(path))
    counts: dict[tuple[int, int], int] = {}
    for t in lib.top_level():
        for p in t.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            counts[(p.layer, p.datatype)] = counts.get((p.layer, p.datatype), 0) + 1
    return counts
