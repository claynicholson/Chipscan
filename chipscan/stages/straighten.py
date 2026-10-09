"""Step 1: straighten the mosaic and remove stitching artefacts.

Two things go wrong between the stage and the stitched mosaic:

* **tilt** - the chip is never perfectly square to the stage.  0.05 deg sounds like nothing,
  but across a 51k px reticle it is a 50 px drift, enough to break "components in a line"
  and every axis-aligned assumption downstream.  We measure it from the image alone:
  rotate the low-res level by candidate angles and keep the one where edge projections
  (column sums of |d/dx|, row sums of |d/dy|) are sharpest - Manhattan layouts make
  this peak very clearly.
* **tile seams** - each camera frame carries its own vignetting and exposure, so a mosaic
  can show a grid of brightness steps.  We look for that grid in the background (FFT of
  the substrate-only profile), fold the image on it to estimate the per-frame shading
  field, and fit one gain per frame.  When no grid is found (a well-stitched scan) nothing
  is changed and the report says so.

The output is a new pyramid plus the exact transform back to the original pixels, so
re-imaging coordinates (PRECiV positions) still refer to the raw scan.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource

warnings.filterwarnings("ignore", "Mean of empty slice")


# ----------------------------------------------------------------------------- tilt
def _sharpness(img: np.ndarray, deg: float) -> float:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    r = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    c = r[h // 8 : -h // 8, w // 8 : -w // 8]
    ex = np.abs(cv2.Sobel(c, cv2.CV_32F, 1, 0))
    ey = np.abs(cv2.Sobel(c, cv2.CV_32F, 0, 1))
    px, py = ex.sum(0), ey.sum(1)
    return float(np.var(px) / max(px.mean(), 1e-6) ** 2 + np.var(py) / max(py.mean(), 1e-6) ** 2)


def estimate_tilt(source: ImageSource, max_deg: float = 2.0, work_px: int = 6000) -> dict:
    """Rotation (deg, OpenCV sign: positive = counter-clockwise) that makes the layout axis-aligned."""
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > work_px:
        level += 1
    img = source.read_level(level).astype(np.float32)
    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.05)
    sc = np.array([_sharpness(img, a) for a in coarse])
    a0 = float(coarse[int(np.argmax(sc))])
    fine = np.arange(a0 - 0.06, a0 + 0.0601, 0.0025)
    sf = np.array([_sharpness(img, a) for a in fine])
    i = int(np.argmax(sf))
    a = float(fine[i])
    if 0 < i < len(sf) - 1:  # parabolic peak refinement
        d = sf[i - 1] - 2 * sf[i] + sf[i + 1]
        if d < 0:
            a += 0.0025 * 0.5 * (sf[i - 1] - sf[i + 1]) / d
    s0 = _sharpness(img, 0.0)
    return {"angle_deg": a, "sharpness_gain": float(sf.max() / max(s0, 1e-9)), "level": level}


# ----------------------------------------------------------------------------- tile seams
@dataclass
class TileGrid:
    found: bool = False
    period_px: tuple[float, float] = (0.0, 0.0)  # (x, y) at level 0
    explained: tuple[float, float] = (0.0, 0.0)  # fraction of background variance the fold explains
    shading: list | None = None  # (ny, nx) multiplicative field over one frame, level-0 bins of `bin_px`
    bin_px: int = 8
    gains: dict = field(default_factory=dict)  # "row,col" -> gain
    amplitude: float = 0.0  # peak-to-peak of the shading field


def _background(img: np.ndarray, f: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Substrate-only background ratio map (nan where structure) at the working level.

    Normalised by a box mean several frames wide, so frame-scale shading survives."""
    bg = ndi.median_filter(img, size=5)
    med = float(np.median(bg))
    sd = np.sqrt(np.maximum(cv2.blur(img * img, (5, 5)) - cv2.blur(img, (5, 5)) ** 2, 0))
    mask = (np.abs(bg - med) < max(6.0, 0.05 * med)) & (sd < np.percentile(sd, 60))
    mask = ndi.binary_erosion(mask, iterations=2)
    k = int(4000 / f) | 1
    num = cv2.blur(np.where(mask, img, 0).astype(np.float32), (k, k))
    den = cv2.blur(mask.astype(np.float32), (k, k))
    big = num / np.maximum(den, 1e-3)
    ratio = np.where(mask & (den > 0.05), img / np.maximum(big, 1.0), np.nan)
    return ratio, mask


def _fold(prof: np.ndarray, f: int, period: float, nb: int) -> tuple[float, np.ndarray]:
    ok = ~np.isnan(prof)
    if ok.sum() < 10:
        return 0.0, np.ones(nb)
    x = np.arange(len(prof)) * f
    b = ((x % period) / period * nb).astype(int) % nb
    sums = np.bincount(b[ok], prof[ok], nb)
    cnt = np.bincount(b[ok], None, nb)
    m = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    m = np.where(np.isnan(m), np.nanmean(m), m)
    expl = float(np.var(m[b[ok]]) / max(np.nanvar(prof[ok]), 1e-12))
    return expl, m


@np.errstate(all="ignore")
def _fold_cv(prof: np.ndarray, f: int, period: float, nb: int) -> float:
    """Cross-validated fold: shading learnt on even frames must explain the odd ones (and vice versa).

    A plain fold overfits - 2P has twice the bins and soaks up random per-frame exposure -
    whereas the vignetting of a real camera frame repeats in every frame."""
    ok = ~np.isnan(prof)
    if ok.sum() < 20:
        return 0.0
    x = np.arange(len(prof)) * f
    b = ((x % period) / period * nb).astype(int) % nb
    fr = (x // period).astype(int)
    par = fr % 2
    # remove each frame's own mean: exposure differs per frame, the vignetting *shape* repeats
    fm = np.bincount(fr[ok], prof[ok]) / np.maximum(np.bincount(fr[ok]), 1)
    prof = prof - fm[np.minimum(fr, len(fm) - 1)]
    sse = sst = 0.0
    for tr in (0, 1):
        a, t = ok & (par == tr), ok & (par != tr)
        if a.sum() < nb or t.sum() < nb:
            return 0.0
        cnt = np.bincount(b[a], None, nb)
        m = np.bincount(b[a], prof[a], nb) / np.maximum(cnt, 1)
        m = np.where(cnt > 0, m, np.nan)
        mu = 0.0
        pred = m[b[t]]
        good = np.isfinite(pred)
        sse += float(((prof[t][good] - pred[good]) ** 2).sum())
        sst += float(((prof[t][good] - mu) ** 2).sum())
    return max(0.0, 1.0 - sse / max(sst, 1e-12))


def estimate_tile_grid(source: ImageSource, work_px: int = 13000, min_explained: float = 0.3, min_amplitude: float = 0.01) -> TileGrid:
    """Find the camera-frame grid from background shading; empty TileGrid when the mosaic is clean."""
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > work_px:
        level += 1
    f = 2**level
    img = source.read_level(level).astype(np.float32)
    ratio, _ = _background(img, f)
    grid = TileGrid(bin_px=max(8, f))
    periods, expls = [], []
    for axis in (0, 1):  # axis 0 -> profile along x
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(ratio, axis=axis) if np.isfinite(ratio).any() else np.full(ratio.shape[1 - axis], np.nan)
        if not np.isfinite(prof).any():
            return grid
        # scan every period (0.25% steps); 2P always explains at least what P does, so take the
        # shortest period that explains nearly the maximum - that is the camera frame
        hi = min(5000.0, len(prof) * f / 4)
        Ps = 300.0 * (1.0025 ** np.arange(int(np.log(hi / 300.0) / np.log(1.0025)) + 1)) if hi > 300 else np.array([])
        if not len(Ps):
            return grid
        ex = np.array([_fold_cv(prof, f, P, max(8, int(P / grid.bin_px))) for P in Ps])
        k = int(np.argmax(ex >= 0.8 * ex.max()))
        k2 = min(len(Ps), k + 40)
        k = k + int(np.argmax(ex[k:k2]))  # local peak just past the threshold crossing
        best = (float(ex[k]), float(Ps[k]))
        expls.append(best[0])
        periods.append(best[1])
    grid.period_px = (periods[0], periods[1])
    grid.explained = (expls[0], expls[1])
    if max(expls) < min_explained:
        return grid
    # 2-D shading field over one frame (separable estimate is robust with sparse background)
    px_, py_ = periods
    nx, ny = max(4, int(px_ / grid.bin_px)), max(4, int(py_ / grid.bin_px))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_x, prof_y = np.nanmean(ratio, axis=0), np.nanmean(ratio, axis=1)
    _, mx = _fold(prof_x, f, px_, nx) if expls[0] >= min_explained else (0, np.ones(nx))
    _, my = _fold(prof_y, f, py_, ny) if expls[1] >= min_explained else (0, np.ones(ny))
    field_ = np.outer(my / np.mean(my), mx / np.mean(mx))
    grid.amplitude = float(field_.max() - field_.min())
    if grid.amplitude < min_amplitude:
        return grid
    grid.shading = field_.astype(np.float32).tolist()
    grid.found = True
    # per-frame gain after flat-fielding: frame median / smooth background
    H, W = img.shape
    for r in range(int(np.ceil(H * f / py_))):
        for c in range(int(np.ceil(W * f / px_))):
            ys, ye = int(r * py_ / f), int(min((r + 1) * py_ / f, H))
            xs, xe = int(c * px_ / f), int(min((c + 1) * px_ / f, W))
            blk = ratio[ys:ye, xs:xe]
            if blk.size and np.isfinite(blk).sum() > 200:
                grid.gains[f"{r},{c}"] = float(np.nanmedian(blk))
    return grid


# ----------------------------------------------------------------------------- apply
@dataclass
class StraightenReport:
    angle_deg: float
    sharpness_gain: float
    tiles: dict
    matrix_out_to_in: list  # 2x3: output (corrected) px -> input (raw) px
    shape: tuple[int, int]


_GAIN_CACHE: dict[int, np.ndarray] = {}


def _gain_table(grid: TileGrid) -> np.ndarray:
    key = id(grid)
    if key not in _GAIN_CACHE:
        rc = [tuple(int(t) for t in k.split(",")) for k in grid.gains]
        G = np.ones((max(r for r, _ in rc) + 1, max(c for _, c in rc) + 1), np.float32)
        for (r, c), v in zip(rc, grid.gains.values(), strict=True):
            G[r, c] = v
        _GAIN_CACHE[key] = G
    return _GAIN_CACHE[key]


def _shading_at(grid: TileGrid, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    fld = np.asarray(grid.shading, np.float32)
    ny, nx = fld.shape
    px_, py_ = grid.period_px
    u = ((xs % px_) / px_ * nx).astype(np.float32)
    v = ((ys % py_) / py_ * ny).astype(np.float32)
    s = cv2.remap(fld, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if grid.gains:
        G = _gain_table(grid)
        r = np.clip((ys // py_).astype(int), 0, G.shape[0] - 1)
        c = np.clip((xs // px_).astype(int), 0, G.shape[1] - 1)
        s = s * G[r, c]
    return s


def straighten(
    source: ImageSource,
    out_path: str | Path,
    angle_deg: float | None = None,
    fix_seams: bool = True,
    min_angle_deg: float = 0.005,
    chunk_px: int = 1024,
    progress=None,
) -> tuple[ImageSource, StraightenReport]:
    """Write a straightened, seam-corrected pyramid to `out_path`; return it and the report."""
    import zarr

    from chipscan.io.pyramid import _build_levels
    from chipscan.io.readers import PyramidSource

    tilt = estimate_tilt(source) if angle_deg is None else {"angle_deg": angle_deg, "sharpness_gain": 1.0}
    a = float(tilt["angle_deg"])
    if abs(a) < min_angle_deg:
        a = 0.0
    grid = estimate_tile_grid(source) if fix_seams else TileGrid()
    H, W = source.shape
    # rotate about the image centre, keep the same canvas: M maps output -> input
    m_in_to_out = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
    m_out_to_in = cv2.invertAffineTransform(m_in_to_out)
    out_path = Path(out_path)
    root = zarr.open_group(str(out_path), mode="w")
    attrs = dict(getattr(source, "attrs", {}) or {})
    attrs.update(
        {
            "pixel_size_um": getattr(source, "pixel_size_um", None),
            "downsample": 2,
            "chunk_px": chunk_px,
            "straighten": {"angle_deg": a, "matrix_out_to_in": m_out_to_in.tolist(), "seams_fixed": bool(grid.found)},
        }
    )
    root.attrs.update(attrs)
    lvl0 = root.create_array("0", shape=(H, W), chunks=(chunk_px, chunk_px), dtype="uint8")
    blocks = [(y, x) for y in range(0, H, chunk_px) for x in range(0, W, chunk_px)]
    margin = int(np.ceil(abs(np.sin(np.radians(a))) * max(H, W))) + 4
    for i, (y, x) in enumerate(blocks):
        bh, bw = min(chunk_px, H - y), min(chunk_px, W - x)
        if a == 0.0 and not grid.found:
            lvl0[y : y + bh, x : x + bw] = source.read_region(x, y, x + bw, y + bh)
        else:
            # input window that covers this output block
            corners = np.array([[x, y, 1], [x + bw, y, 1], [x, y + bh, 1], [x + bw, y + bh, 1]], np.float64) @ m_out_to_in.T
            ix0, iy0 = int(np.floor(corners[:, 0].min())) - 2, int(np.floor(corners[:, 1].min())) - 2
            ix1, iy1 = int(np.ceil(corners[:, 0].max())) + 3, int(np.ceil(corners[:, 1].max())) + 3
            src = source.read_region(ix0, iy0, ix1, iy1).astype(np.float32)
            if grid.found:
                yy, xx = np.mgrid[iy0:iy1, ix0:ix1].astype(np.float32)
                src = src / np.maximum(_shading_at(grid, xx, yy), 0.2)
            # output block pixel (u, v) -> input (m @ [x+u, y+v, 1]) - (ix0, iy0)
            m = m_out_to_in.copy()
            m[:, 2] = m_out_to_in @ np.array([x, y, 1.0]) - np.array([ix0, iy0])
            out = cv2.warpAffine(src, m, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            lvl0[y : y + bh, x : x + bw] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        if progress:
            progress(i + 1, len(blocks))
    del margin
    _build_levels(root, chunk_px, 512)
    root.attrs["levels"] = len(list(root.array_keys()))
    g = asdict(grid)
    g.pop("shading", None)
    report = StraightenReport(a, float(tilt.get("sharpness_gain", 1.0)), g, m_out_to_in.tolist(), (H, W))
    (out_path.parent / (out_path.stem + ".straighten.json")).write_text(json.dumps(asdict(report), indent=2))
    return PyramidSource(out_path), report


def to_raw_px(matrix_out_to_in: list, x: float, y: float) -> tuple[float, float]:
    """Corrected-image px -> raw-scan px (for re-imaging positions)."""
    m = np.asarray(matrix_out_to_in, np.float64)
    p = m @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])
