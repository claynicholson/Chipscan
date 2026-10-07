"""Die-to-die detection (design-free): compare each region with the median of its twins.

Chips on a reticle repeat at a known pitch (estimated by autocorrelation, see repeats.py).
For every tile we read the same tile from the neighbouring chips, align each twin by
phase correlation (stitching and stage error are a few px), take the per-pixel median
across twins, and flag pixels where the tile departs from that median by more than
`z_min` noise sigmas *and* by more than the twins disagree among themselves (so design
differences between parameter splits do not fire).  No model, no training, no GDS: a
defect is simply something that is not on the other chips.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from chipscan.io.readers import ImageSource

_LOWRES_CACHE: dict[int, tuple[int, np.ndarray]] = {}


def _lowres(source: ImageSource, level: int) -> tuple[int, np.ndarray]:
    key = id(source)
    if key not in _LOWRES_CACHE or _LOWRES_CACHE[key][0] != level:
        _LOWRES_CACHE[key] = (level, source.read_level(level).astype(np.float32))
    return _LOWRES_CACHE[key]


def find_twins(
    source: ImageSource, x: int, y: int, size: int, level: int = 3, min_ncc: float = 0.65, max_twins: int = 8, min_dist_px: int = 1200
) -> list[tuple[int, int, float]]:
    """Pitch-free twin search: template-match the tile (at `level`) along its row and its column.

    Returns [(x, y, ncc)] of the strongest matches at least `min_dist_px` away. Chips on a reticle
    repeat along x; arrays and rows of structures repeat along y; both are searched.
    """
    import cv2

    s = 2**level
    _, low = _lowres(source, level)
    h, w = low.shape
    tx0, ty0, ts = x // s, y // s, max(8, size // s)
    tile = low[ty0 : ty0 + ts, tx0 : tx0 + ts]
    if tile.shape[0] < 8 or tile.shape[1] < 8 or tile.std() < 1.0:
        return []
    cands: list[tuple[int, int, float]] = []
    row = low[ty0 : ty0 + ts, :]
    if row.shape[1] > ts + 1:
        r = cv2.matchTemplate(row, tile, cv2.TM_CCOEFF_NORMED)[0]
        for i in np.argsort(r)[::-1][:200]:
            if r[i] < min_ncc:
                break
            if abs(i - tx0) * s >= min_dist_px and all(abs(i - c[0] // s) * s >= min_dist_px for c in cands):
                cands.append((int(i * s), y, float(r[i])))
    col = low[:, tx0 : tx0 + ts]
    if col.shape[0] > ts + 1:
        c = cv2.matchTemplate(col, tile, cv2.TM_CCOEFF_NORMED)[:, 0]
        for j in np.argsort(c)[::-1][:200]:
            if c[j] < min_ncc:
                break
            if abs(j - ty0) * s >= min_dist_px and all(abs(j * s - cc[1]) >= min_dist_px or cc[0] != x for cc in cands):
                cands.append((x, int(j * s), float(c[j])))
    cands.sort(key=lambda t: -t[2])
    return cands[:max_twins]


def _align(ref: np.ndarray, mov: np.ndarray, max_shift: int = 12) -> np.ndarray | None:
    if ref.std() < 1e-3 or mov.std() < 1e-3:
        return mov
    try:
        s, _, _ = phase_cross_correlation(ref, mov, upsample_factor=4, normalization="phase")
    except Exception:
        return None
    if abs(s[0]) > max_shift or abs(s[1]) > max_shift:
        return None
    return ndi.shift(mov, s, order=1, mode="nearest")


def die2die_tile(
    source: ImageSource,
    x: int,
    y: int,
    size: int,
    pitches: dict[str, list[float]],
    z_min: float = 5.0,
    min_twins: int = 3,
    smooth: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Returns (mask, zmap, n_twins) for the tile at (x, y); None if too few twins."""
    H, W = source.shape
    img = source.read_region(x, y, x + size, y + size).astype(np.float32) / 255.0
    tw = []
    for tx, ty, _ncc in find_twins(source, x, y, size):
        if tx < 0 or ty < 0 or tx + size > W or ty + size > H:
            continue
        t = source.read_region(tx, ty, tx + size, ty + size).astype(np.float32) / 255.0
        a = _align(img, t, max_shift=max(12, 2 ** 3))
        if a is not None:
            tw.append(a)
    if len(tw) < min_twins:
        return None, None, len(tw)
    stack = np.stack(tw)
    med = np.median(stack, axis=0)
    spread = np.median(np.abs(stack - med), axis=0) * 1.4826 + 1e-3  # how much the twins disagree
    diff = ndi.gaussian_filter(img - med, smooth) if smooth > 0 else img - med
    noise = max(1.4826 * float(np.median(np.abs(diff - np.median(diff)))), 1.0 / 255.0)
    z = np.abs(diff) / noise
    # must also exceed the twins' own disagreement (design splits, residual misalignment at edges)
    z[np.abs(diff) < 3.0 * ndi.gaussian_filter(spread, smooth)] = 0.0
    return z > z_min, z, len(tw)


def die2die(
    source: ImageSource,
    pitches: dict[str, list[float]],
    tile_px: int = 1024,
    margin_px: int = 32,
    z_min: float = 5.0,
    min_area_px: int = 6,
    merge_px: int = 6,
    pixel_size_um: float | None = None,
    region=None,
    id_start: int = 0,
    progress: bool = True,
) -> list[dict]:
    import cv2
    from rich.progress import Progress

    from chipscan.tiling import make_tiles

    H, W = source.shape
    tiles = make_tiles(source.shape, tile_px, margin_px, region)
    recs: list[dict] = []
    nid = id_start
    prog = Progress() if progress else None
    if prog:
        prog.start()
        task = prog.add_task("die-to-die", total=len(tiles))
    px = pixel_size_um or 1.0
    for t in tiles:
        x0, y0, x1, y1 = t.read.to_int()
        x0, y0 = max(x0, 0), max(y0, 0)
        size = min(x1, W) - x0, min(y1, H) - y0
        # square read so twins read identically
        s = min(size)
        mask, z, n = die2die_tile(source, x0, y0, s, pitches, z_min)
        if mask is not None:
            cy0, cx0 = int(t.core.y0) - y0, int(t.core.x0) - x0
            core = mask[cy0 : cy0 + int(t.core.height), cx0 : cx0 + int(t.core.width)]
            zc = z[cy0 : cy0 + int(t.core.height), cx0 : cx0 + int(t.core.width)]
            if core.any():
                grown = ndi.binary_dilation(core, iterations=merge_px) if merge_px else core
                lab_g, _ = ndi.label(grown)
                lab = lab_g * core
                for i, sl in enumerate(ndi.find_objects(lab), start=1):
                    if sl is None:
                        continue
                    comp = lab[sl] == i
                    area = int(comp.sum())
                    if area < min_area_px:
                        continue
                    cy, cx = ndi.center_of_mass(comp)
                    gx, gy = cx + sl[1].start + int(t.core.x0), cy + sl[0].start + int(t.core.y0)
                    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polys = [(c.reshape(-1, 2).astype(float) + [sl[1].start + t.core.x0 + 0.5, sl[0].start + t.core.y0 + 0.5]).round(2).tolist() for c in cs if len(c) >= 3]
                    bb = {"x0": float(sl[1].start + t.core.x0), "y0": float(sl[0].start + t.core.y0), "x1": float(sl[1].stop + t.core.x0), "y1": float(sl[0].stop + t.core.y0)}
                    recs.append(
                        {
                            "id": nid,
                            "kind": "die2die",
                            "class": "die2die",
                            "area_px": area,
                            "score": float(min(1.0, zc[sl][comp].max() / 20.0)),
                            "z": float(zc[sl][comp].max()),
                            "n_twins": n,
                            "bbox_px": bb,
                            "centroid_px": [float(gx), float(gy)],
                            "polygons_px": polys,
                            "area_um2": area * px**2,
                            "centroid_um": [float(gx) * px, float(gy) * px],
                            "bbox_um": {k: v * px for k, v in bb.items()},
                        }
                    )
                    nid += 1
        if prog:
            prog.advance(task)
    if prog:
        prog.stop()
    return recs
