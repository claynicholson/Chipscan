"""Coordinate frames and transforms.

Two frames exist everywhere in chipscan:

* **world**: the design frame, in microns, as drawn in the GDS.
* **px**: level-0 pixel coordinates of the image, (x right, y down).

An :class:`Affine2D` maps world -> px.  A :class:`WarpField` adds a smooth local
residual on top of a global affine so that stitching distortion in a mosaic
can be absorbed without giving up a closed-form global model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box, half-open on the max side, any unit."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def corners(self) -> np.ndarray:
        return np.array(
            [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]],
            dtype=np.float64,
        )

    def intersection(self, other: BBox) -> BBox | None:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(x0, y0, x1, y1)

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def overlaps(self, other: BBox, margin: float = 0.0) -> bool:
        return not (
            self.x1 + margin <= other.x0
            or other.x1 + margin <= self.x0
            or self.y1 + margin <= other.y0
            or other.y1 + margin <= self.y0
        )

    def pad(self, m: float) -> BBox:
        return BBox(self.x0 - m, self.y0 - m, self.x1 + m, self.y1 + m)

    def to_int(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) as ints covering the box (floor/ceil)."""
        return (
            int(math.floor(self.x0)),
            int(math.floor(self.y0)),
            int(math.ceil(self.x1)),
            int(math.ceil(self.y1)),
        )

    def clip(self, other: BBox) -> BBox:
        inter = self.intersection(other)
        return inter if inter is not None else BBox(self.x0, self.y0, self.x0, self.y0)

    @staticmethod
    def of_points(pts: np.ndarray) -> BBox:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return BBox(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


class Affine2D:
    """2-D affine transform stored as a 3x3 homogeneous matrix.

    By convention in chipscan an Affine2D maps *world (um)* to *px*.
    """

    __slots__ = ("m",)

    def __init__(self, m: np.ndarray | None = None):
        self.m = np.eye(3, dtype=np.float64) if m is None else np.asarray(m, dtype=np.float64)
        assert self.m.shape == (3, 3)

    # ---- constructors -------------------------------------------------
    @staticmethod
    def identity() -> Affine2D:
        return Affine2D()

    @staticmethod
    def from_similarity(
        scale: float,
        rotation_deg: float,
        tx: float,
        ty: float,
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> Affine2D:
        """scale * R(rotation) * F(flip) then translate."""
        th = math.radians(rotation_deg)
        c, s = math.cos(th), math.sin(th)
        f = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0, 1.0])
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        sm = np.diag([scale, scale, 1.0])
        t = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        return Affine2D(t @ sm @ r @ f)

    @staticmethod
    def translation(tx: float, ty: float) -> Affine2D:
        return Affine2D(np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]))

    @staticmethod
    def scaling(sx: float, sy: float | None = None) -> Affine2D:
        sy = sx if sy is None else sy
        return Affine2D(np.diag([sx, sy, 1.0]))

    @staticmethod
    def fit(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> Affine2D:
        """Least-squares affine mapping src (N,2) -> dst (N,2)."""
        src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
        n = len(src)
        if n < 3:
            raise ValueError("need at least 3 point pairs to fit an affine")
        a = np.hstack([src, np.ones((n, 1))])
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        sw = np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(a * sw, dst * sw, rcond=None)
        m = np.eye(3)
        m[:2, :] = sol.T
        return Affine2D(m)

    # ---- algebra ------------------------------------------------------
    def compose(self, inner: Affine2D) -> Affine2D:
        """Return self ∘ inner  (apply `inner` first, then `self`)."""
        return Affine2D(self.m @ inner.m)

    def __matmul__(self, inner: Affine2D) -> Affine2D:
        return self.compose(inner)

    def inverse(self) -> Affine2D:
        return Affine2D(np.linalg.inv(self.m))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        shp = pts.shape
        p = pts.reshape(-1, 2)
        out = p @ self.m[:2, :2].T + self.m[:2, 2]
        return out.reshape(shp)

    def apply_bbox(self, b: BBox) -> BBox:
        return BBox.of_points(self.apply(b.corners()))

    # ---- decomposition -----------------------------------------------
    @property
    def linear(self) -> np.ndarray:
        return self.m[:2, :2]

    @property
    def translation_xy(self) -> tuple[float, float]:
        return (float(self.m[0, 2]), float(self.m[1, 2]))

    @property
    def scale_factor(self) -> float:
        """Geometric mean scale (px per world unit)."""
        return float(math.sqrt(abs(np.linalg.det(self.linear))))

    @property
    def is_flipped(self) -> bool:
        return bool(np.linalg.det(self.linear) < 0)

    @property
    def rotation_deg(self) -> float:
        a = self.linear.copy()
        if self.is_flipped:
            a = a @ np.diag([-1.0, 1.0])
        return float(math.degrees(math.atan2(a[1, 0], a[0, 0])))

    # ---- io -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.m.tolist(),
            "scale": self.scale_factor,
            "rotation_deg": self.rotation_deg,
            "flipped": self.is_flipped,
            "translation": self.translation_xy,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Affine2D:
        return Affine2D(np.array(d["matrix"], dtype=np.float64))

    def __repr__(self) -> str:
        return (
            f"Affine2D(scale={self.scale_factor:.6g}, rot={self.rotation_deg:.4f}deg, "
            f"flip={self.is_flipped}, t={self.translation_xy})"
        )


class WarpField:
    """Global affine plus a smooth residual shift field sampled on a px grid.

    ``world_to_px(p) = A(p) + R(A(p))`` where R is bilinearly interpolated from
    ``(grid_x, grid_y, dx, dy)``.  Outside the grid R is clamped to the edge.
    Residuals are what is left after the best global affine: stitching seams,
    stage non-linearity, slight optical distortion.
    """

    def __init__(
        self,
        affine: Affine2D,
        grid_x: np.ndarray | None = None,
        grid_y: np.ndarray | None = None,
        dx: np.ndarray | None = None,
        dy: np.ndarray | None = None,
    ):
        self.affine = affine
        self.grid_x = None if grid_x is None else np.asarray(grid_x, dtype=np.float64)
        self.grid_y = None if grid_y is None else np.asarray(grid_y, dtype=np.float64)
        self.dx = None if dx is None else np.asarray(dx, dtype=np.float64)
        self.dy = None if dy is None else np.asarray(dy, dtype=np.float64)
        self._interp_x = self._interp_y = None
        if self.has_residual:
            self._interp_x = RegularGridInterpolator(
                (self.grid_y, self.grid_x), self.dx, bounds_error=False, fill_value=None
            )
            self._interp_y = RegularGridInterpolator(
                (self.grid_y, self.grid_x), self.dy, bounds_error=False, fill_value=None
            )

    @property
    def has_residual(self) -> bool:
        return (
            self.grid_x is not None
            and self.grid_y is not None
            and self.dx is not None
            and self.dx.size > 0
            and len(self.grid_x) >= 2
            and len(self.grid_y) >= 2
        )

    def residual_at_px(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        if not self.has_residual:
            return np.zeros_like(px)
        # clamp to the grid so edge tiles extrapolate flat rather than linearly
        q = np.column_stack(
            [
                np.clip(px[:, 1], self.grid_y[0], self.grid_y[-1]),
                np.clip(px[:, 0], self.grid_x[0], self.grid_x[-1]),
            ]
        )
        return np.column_stack([self._interp_x(q), self._interp_y(q)])

    def world_to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        shp = pts.shape
        p = self.affine.apply(pts.reshape(-1, 2))
        p = p + self.residual_at_px(p)
        return p.reshape(shp)

    def px_to_world(self, pts: np.ndarray, iters: int = 3) -> np.ndarray:
        """Invert by fixed-point iteration (residual is small and smooth)."""
        pts = np.asarray(pts, dtype=np.float64)
        shp = pts.shape
        p = pts.reshape(-1, 2)
        inv = self.affine.inverse()
        q = p.copy()
        for _ in range(iters if self.has_residual else 1):
            q = p - self.residual_at_px(q)
        return inv.apply(q).reshape(shp)

    def local_affine(self, px_bbox: BBox) -> Affine2D:
        """Best affine approximation of the warp inside a px box.

        Tile-level code uses this so that rendering stays a single fillPoly call.
        """
        if not self.has_residual:
            return self.affine
        # sample residual at the box corners + centre and fit an affine correction
        c = np.vstack([px_bbox.corners(), [px_bbox.center]])
        corr = Affine2D.fit(c, c + self.residual_at_px(c))
        return corr.compose(self.affine)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"affine": self.affine.to_dict()}
        if self.has_residual:
            d["residual"] = {
                "grid_x": self.grid_x.tolist(),
                "grid_y": self.grid_y.tolist(),
                "dx": self.dx.tolist(),
                "dy": self.dy.tolist(),
            }
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> WarpField:
        aff = Affine2D.from_dict(d["affine"])
        r = d.get("residual")
        if not r:
            return WarpField(aff)
        return WarpField(
            aff, np.array(r["grid_x"]), np.array(r["grid_y"]), np.array(r["dx"]), np.array(r["dy"])
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> WarpField:
        return WarpField.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        res = ""
        if self.has_residual:
            res = f", residual grid {self.dx.shape} max|r|={np.hypot(self.dx, self.dy).max():.2f}px"
        return f"WarpField({self.affine!r}{res})"
