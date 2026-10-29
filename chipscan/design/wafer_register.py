"""Register a wafer scan to its WaferDesign: global affine + a local displacement field.

The stitched mosaic is not one rigid image: camera frames sit a few to tens of px off where a
perfect stage would put them.  After a global affine (scale = the true pixel size), shifts are
measured on a dense grid of tiles (edge-map phase correlation against the rendered design,
kept only where the match is unambiguous), cleaned with a median filter and interpolated to a
smooth field.  `to_px(x_um, y_um)` / `to_um(col, row)` then map with the field applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata

from chipscan.design.wafer import WaferDesign
from chipscan.io.readers import ImageSource
from chipscan.registration.coarse import edge_map, ncc


@dataclass
class WaferRegistration:
    affine: np.ndarray  # 2x3: [col,row] = A @ [x_um, y_um, 1]
    field: np.ndarray  # (gh, gw, 2) px correction (dcol, drow) on a grid
    step: int  # grid spacing in level-0 px
    score: float = 0.0
    frames: np.ndarray | None = None  # (fh, fw, 2) per camera frame offset (stitching), or None
    frame_period: tuple[float, float] | None = None  # (x, y) camera frame size in px

    @property
    def pixel_size_um(self) -> float:
        return float(1.0 / np.sqrt(abs(np.linalg.det(self.affine[:, :2]))))

    def correction(self, col, row):
        shape = np.shape(col)
        c = np.atleast_1d(np.asarray(col, np.float64)).ravel() / self.step
        r = np.atleast_1d(np.asarray(row, np.float64)).ravel() / self.step
        dc = ndi.map_coordinates(self.field[..., 0], [r, c], order=1, mode="nearest").reshape(shape)
        dr = ndi.map_coordinates(self.field[..., 1], [r, c], order=1, mode="nearest").reshape(shape)
        if self.frames is not None and self.frame_period:
            # a stitched mosaic moves frame by frame: piecewise constant on the camera grid
            fx = np.clip((c * self.step / self.frame_period[0]).astype(int), 0, self.frames.shape[1] - 1)
            fy = np.clip((r * self.step / self.frame_period[1]).astype(int), 0, self.frames.shape[0] - 1)
            dc = dc + self.frames[fy, fx, 0].reshape(shape)
            dr = dr + self.frames[fy, fx, 1].reshape(shape)
        return dc, dr

    def to_px(self, x_um, y_um):
        x, y = np.asarray(x_um, np.float64), np.asarray(y_um, np.float64)
        c = self.affine[0, 0] * x + self.affine[0, 1] * y + self.affine[0, 2]
        r = self.affine[1, 0] * x + self.affine[1, 1] * y + self.affine[1, 2]
        dc, dr = self.correction(c, r)
        return c + dc, r + dr

    def render_window(self, wd: WaferDesign, col0: int, row0: int, w: int, h: int, level: int = 0) -> np.ndarray:
        """Design metal (0/1 uint8) for a scan window, with the local correction at its centre."""
        s = 2**level
        dc, dr = self.correction(col0 + w * s / 2, row0 + h * s / 2)
        Ai = np.linalg.inv(np.vstack([self.affine, [0, 0, 1]]))[:2]
        corners = np.array([[col0, row0, 1], [col0 + w * s, row0, 1], [col0, row0 + h * s, 1], [col0 + w * s, row0 + h * s, 1]], np.float64)
        corners[:, 0] -= float(dc)
        corners[:, 1] -= float(dr)
        um = corners @ Ai.T
        pad = 50 * self.pixel_size_um
        bx = (um[:, 0].min() - pad, um[:, 1].min() - pad, um[:, 0].max() + pad, um[:, 1].max() + pad)
        d = wd.render(self.pixel_size_um * s, bx).astype(np.float32)
        oc, orow = self.affine @ np.array([bx[0], bx[3], 1.0])
        M = np.float32([[1, 0, (oc + float(dc) - col0) / s], [0, 1, (orow + float(dr) - row0) / s]])
        return (cv2.warpAffine(d, M, (w, h), flags=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

    def save(self, path: str | Path) -> None:
        extra = {"frames": self.frames, "frame_period": np.asarray(self.frame_period)} if self.frames is not None else {}
        np.savez_compressed(path, affine=self.affine, field=self.field, step=self.step, score=self.score, **extra)

    @staticmethod
    def load(path: str | Path) -> WaferRegistration:
        z = np.load(path)
        fr = z["frames"] if "frames" in z else None
        fp = tuple(map(float, z["frame_period"])) if "frame_period" in z else None
        return WaferRegistration(z["affine"], z["field"], int(z["step"]), float(z["score"]), fr, fp)


_FR: dict = {}


def _frame_init(src_path: str, gds_path: str, reg_path: str) -> None:
    import torch  # noqa: F401

    from chipscan.io.readers import PyramidSource

    _FR.update(src=PyramidSource(src_path), wd=WaferDesign.load(gds_path), reg=WaferRegistration.load(reg_path))


def _frame_shift(args):
    fy, fx, c, r = args
    m = measure_shift(_FR["src"], _FR["wd"], _FR["reg"], c, r, n=1024, level=0, min_ncc=0.35)
    return fy, fx, m


def frame_offsets(src_path: str, gds_path: str, reg_path: str, shape: tuple[int, int], period: tuple[float, float], mask=None, workers: int | None = None, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Residual (dcol, drow) per camera frame, measured on a 1024 px tile in each frame's middle.
    Frames without enough design structure take the median of their neighbours."""
    from concurrent.futures import ProcessPoolExecutor

    from chipscan import default_workers

    H, W = shape
    px, py = period
    fw, fh = int(np.ceil(W / px)), int(np.ceil(H / py))
    jobs = []
    for fy in range(fh):
        for fx in range(fw):
            c, r = int(fx * px + px / 2 - 512), int(fy * py + py / 2 - 512)
            if c < 0 or r < 0 or c + 1024 > W or r + 1024 > H:
                continue
            if mask is not None and not mask.any_in(c, r, c + 1024, r + 1024):
                continue
            jobs.append((fy, fx, c, r))
    out = np.full((fh, fw, 2), np.nan)
    score = np.zeros((fh, fw))
    with ProcessPoolExecutor(max_workers=workers or default_workers(), initializer=_frame_init, initargs=(src_path, gds_path, reg_path)) as ex:
        for i, (fy, fx, m) in enumerate(ex.map(_frame_shift, jobs, chunksize=8)):
            if m is not None:
                out[fy, fx] = m[:2]
                score[fy, fx] = m[2]
            if progress and i % 200 == 0:
                progress(i, len(jobs))
    # robust: a frame far from its neighbours locked onto a wrong period of a repeated structure
    med = np.stack([ndi.generic_filter(out[..., k], np.nanmedian, size=3, mode="nearest") for k in (0, 1)], -1)
    bad = np.hypot(*(out - med).transpose(2, 0, 1)) > 6
    out[bad] = np.nan
    filled = out.copy()
    for _ in range(50):  # fill holes from neighbours
        nanm = np.isnan(filled[..., 0])
        if not nanm.any():
            break
        nb = np.stack([ndi.generic_filter(filled[..., k], np.nanmedian, size=3, mode="nearest") for k in (0, 1)], -1)
        filled[nanm] = nb[nanm]
    filled = np.nan_to_num(filled)
    return filled, score


def measure_shift(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, c: int, r: int, n: int = 1024, level: int = 0, min_ncc: float = 0.3):
    s = 2**level
    c, r = c // s * s, r // s * s
    img = src.read_region(c, r, c + n * s, r + n * s, level=level).astype(np.float32)[:n, :n] / 255.0
    if img.std() < 0.03:
        return None
    d = reg.render_window(wd, c, r, n, n, level).astype(np.float32)
    if d.std() < 0.05:
        return None
    from skimage.registration import phase_cross_correlation

    ie, de = edge_map(img, 1.2), edge_map(d, 1.2)
    sh, _, _ = phase_cross_correlation(ie, de, upsample_factor=10, normalization="phase")
    if abs(sh[0]) > n / 4 or abs(sh[1]) > n / 4:
        return None
    d2 = cv2.warpAffine(d, np.float32([[1, 0, sh[1]], [0, 1, sh[0]]]), (n, n))
    sc = ncc(ie, edge_map(d2, 1.2))
    if sc < min_ncc:
        return None
    return float(sh[1] * s), float(sh[0] * s), float(sc)


def refine_field(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, step: int = 2048, level: int = 0, mask=None, progress=None, workers: int = 1) -> WaferRegistration:
    """Measure residual shifts on a grid and fold them into the field."""
    H, W = src.shape
    gh, gw = -(-H // step) + 1, -(-W // step) + 1
    pts, vals = [], []
    cells = [(gy, gx) for gy in range(gh) for gx in range(gw)]
    for i, (gy, gx) in enumerate(cells):
        c, r = gx * step - 512 * 2**level, gy * step - 512 * 2**level
        if c < 0 or r < 0 or c + 1024 * 2**level > W or r + 1024 * 2**level > H:
            continue
        if mask is not None and not mask.any_in(c, r, c + 1024, r + 1024):
            continue
        m = measure_shift(src, wd, reg, c, r, level=level)
        if m is not None:
            pts.append((gy, gx))
            vals.append(m[:2])
        if progress and i % 50 == 0:
            progress(i, len(cells))
    pts, vals = np.array(pts), np.array(vals)
    new = np.zeros((gh, gw, 2))
    if len(pts) >= 4:
        # outliers (a tile locked onto the wrong period) out: compare with the neighbourhood median
        grid = np.full((gh, gw, 2), np.nan)
        grid[pts[:, 0], pts[:, 1]] = vals
        med = np.stack([ndi.generic_filter(grid[..., k], np.nanmedian, size=5, mode="nearest") for k in (0, 1)], -1)
        ok = np.hypot(*(grid[pts[:, 0], pts[:, 1]] - med[pts[:, 0], pts[:, 1]]).T) < 6 * 2**level + 4
        pts, vals = pts[ok], vals[ok]
        gy, gx = np.mgrid[0:gh, 0:gw]
        for k in (0, 1):
            f = griddata(pts, vals[:, k], (gy, gx), method="linear")
            f2 = griddata(pts, vals[:, k], (gy, gx), method="nearest")
            new[..., k] = np.where(np.isnan(f), f2, f)
    # the previous field, resampled onto this grid (it may have had another spacing)
    gy, gx = np.mgrid[0:gh, 0:gw]
    old = np.stack(reg.correction(gx * step, gy * step), -1)
    out = WaferRegistration(reg.affine, old + new, step, reg.score)
    return out, len(pts)
