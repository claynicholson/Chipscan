"""Local (fine) registration: refine the global affine and recover a residual warp.

Sample an N x N grid of tiles over the image/design overlap.  In each tile,
phase-correlate the image edge map with the rendered design edge map to get a
local shift.  Then:

* refit the global affine to all confident tile shifts (robust least squares), and
* keep what is left over as a smooth residual field (:class:`WarpField`) so that
  stitching seams and stage non-linearity do not become false defects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.registration import phase_cross_correlation

from chipscan.config import FineRegistrationConfig
from chipscan.design import Design
from chipscan.geometry import Affine2D, BBox, WarpField
from chipscan.io.readers import ImageSource
from chipscan.registration.coarse import edge_map, ncc
from chipscan.tiling import grid_sample_boxes


@dataclass
class FineResult:
    warp: WarpField
    tiles: list[dict] = field(default_factory=list)
    rms_residual_px: float = 0.0
    n_used: int = 0
    n_total: int = 0

    def to_dict(self) -> dict:
        return {
            "warp": self.warp.to_dict(),
            "rms_residual_px": self.rms_residual_px,
            "n_used": self.n_used,
            "n_total": self.n_total,
            "tiles": self.tiles,
        }


def _tile_shift(img: np.ndarray, des: np.ndarray, max_shift: float, sigma: float = 1.0) -> tuple[float, float, float]:
    """Returns (dx, dy, confidence) in the tile's pixel units; shift moves design onto image."""
    ie, de = edge_map(img, sigma), edge_map(des, sigma)
    if ie.shape != de.shape:  # defensive: crop to the common shape rather than failing silently
        h, w = min(ie.shape[0], de.shape[0]), min(ie.shape[1], de.shape[1])
        ie, de = ie[:h, :w], de[:h, :w]
    if de.max() == 0 or ie.std() < 1e-3 or (de > 0.2).mean() < 0.002:
        return 0.0, 0.0, 0.0
    try:
        shift, _, _ = phase_cross_correlation(ie, de, upsample_factor=10, normalization="phase")
    except Exception:  # degenerate tile (all-constant edge map etc.)
        return 0.0, 0.0, 0.0
    dy, dx = float(shift[0]), float(shift[1])
    if not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0, 0.0
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return 0.0, 0.0, 0.0
    # confidence: NCC after applying the integer part of the shift
    rolled = np.roll(np.roll(de, int(round(dy)), axis=0), int(round(dx)), axis=1)
    inner = (slice(int(max_shift), -int(max_shift) or None), slice(int(max_shift), -int(max_shift) or None))
    conf = ncc(ie[inner], rolled[inner])
    conf0 = ncc(ie[inner], de[inner])
    return dx, dy, max(conf, 0.0) if conf >= conf0 - 0.02 else 0.0


def fine_register(
    source: ImageSource,
    design: Design,
    coarse: Affine2D,
    cfg: FineRegistrationConfig,
    pixel_size_um: float,
) -> FineResult:
    level = 0
    while level + 1 < source.num_levels and pixel_size_um * 2 ** (level + 1) <= cfg.level_pixel_size_um:
        level += 1
    s = 2**level
    overlap = coarse.apply_bbox(design.bbox).intersection(source.bbox_px)
    if overlap is None:
        raise RuntimeError("design does not overlap the image after coarse registration")
    boxes = grid_sample_boxes(overlap, cfg.grid, cfg.tile_px * s)
    to_lvl = Affine2D.scaling(1.0 / s)
    a_lvl = to_lvl.compose(coarse)  # world -> level px

    centers, dsts, confs, recs = [], [], [], []
    for b in boxes:
        x0, y0, _, _ = b.to_int()
        # snap to the pyramid level's grid so image tile and design render have identical shapes
        x0, y0 = x0 - x0 % s, y0 - y0 % s
        x1, y1 = x0 + cfg.tile_px * s, y0 + cfg.tile_px * s
        img = source.read_region(x0, y0, x1, y1, level=level)
        lb = BBox(x0 // s, y0 // s, x0 // s + img.shape[1], y0 // s + img.shape[0])
        des = design.render(a_lvl, lb)
        dx, dy, conf = _tile_shift(img, des, cfg.max_shift_px)
        c = np.array(b.center)
        rec = {"cx": float(c[0]), "cy": float(c[1]), "dx": dx * s, "dy": dy * s, "conf": conf}
        recs.append(rec)
        if conf >= cfg.min_confidence:
            centers.append(c)
            dsts.append(c + np.array([dx * s, dy * s]))
            confs.append(conf)

    n_total = len(boxes)
    if len(centers) < 4:
        return FineResult(WarpField(coarse), recs, 0.0, len(centers), n_total)

    centers_a = np.array(centers)
    dsts_a = np.array(dsts)
    w = np.array(confs)
    world = coarse.inverse().apply(centers_a)

    # robust refit: drop tiles whose residual is far from the median
    keep = np.ones(len(world), dtype=bool)
    aff = coarse
    for _ in range(3):
        aff = Affine2D.fit(world[keep], dsts_a[keep], w[keep])
        res = np.linalg.norm(aff.apply(world) - dsts_a, axis=1)
        med = np.median(res[keep])
        mad = np.median(np.abs(res[keep] - med)) + 1e-6
        new_keep = res <= med + 3.5 * mad + 0.5
        if new_keep.sum() < 4 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    resid = dsts_a - aff.apply(world)
    rms = float(np.sqrt((resid[keep] ** 2).sum(axis=1).mean()))

    warp = WarpField(aff)
    if cfg.residual_field:
        # lay residuals on the sampling grid; unknown tiles get 0
        gx = np.unique(np.round([b.center[0] for b in boxes], 3))
        gy = np.unique(np.round([b.center[1] for b in boxes], 3))
        if len(gx) >= 2 and len(gy) >= 2:
            dxg = np.zeros((len(gy), len(gx)))
            dyg = np.zeros((len(gy), len(gx)))
            for i, (c, ok) in enumerate(zip(centers_a, keep, strict=True)):
                if not ok:
                    continue
                r = int(np.argmin(np.abs(gy - round(c[1], 3))))
                q = int(np.argmin(np.abs(gx - round(c[0], 3))))
                dxg[r, q], dyg[r, q] = resid[i]
            warp = WarpField(aff, gx, gy, dxg, dyg)

    for rec, ok in zip([r for r in recs if r["conf"] >= cfg.min_confidence], keep, strict=True):
        rec["used"] = bool(ok)
    return FineResult(warp, recs, rms, int(keep.sum()), n_total)
