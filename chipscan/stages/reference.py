"""Reference residual for the whole scan: every tile compared with its twins.

Chips on a reticle and cells in an array repeat.  For each tile we find its twins anywhere on
the same row band or column band (pitch-free template matching at a coarse level, +-48 px
slack for stage placement), align each twin sub-pixel, match its brightness, and build a 1 px
tolerance envelope (grey erosion/dilation).  The median over twins of how far the tile falls
outside each envelope is the residual: zero for anything designed (every twin has it), large
for anything only this location has - dust, particles, fibres, residue, scratches.

The residual is the highest-recall dirt signal we have that does not depend on what the dirt
looks like.  It is stored as a uint8 map (residual x 1000, clipped) plus the number of twins per
tile (0 = unique structure, no reference: other detectors must cover it).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from chipscan import default_workers
from scipy import ndimage as ndi

SCALE = 1000.0  # residual (0..1 grey) -> uint8 units; 0.255 saturates


def _tile_residual(args) -> tuple[int, int, np.ndarray, int]:
    path, x0, y0, size, margin = args
    from chipscan.io.readers import PyramidSource
    from chipscan.stages.verify import _shift, find_twins_slack

    src = _SRC.get(path)
    if src is None:
        src = _SRC[path] = PyramidSource(path)
    H, W = src.shape
    rx0, ry0 = max(0, x0 - margin), max(0, y0 - margin)
    rs = size + 2 * margin
    rx0, ry0 = min(rx0, W - rs), min(ry0, H - rs)
    img = src.read_region(rx0, ry0, rx0 + rs, ry0 + rs).astype(np.float32) / 255.0
    out = np.zeros((size, size), np.uint8)
    if img.std() < 0.004:
        return x0, y0, out, -1
    if src.num_levels <= 3:  # too small to search for twins at the coarse level
        return x0, y0, _fallback(img, x0 - rx0, y0 - ry0, size), 0
    twins = [t for t in find_twins_slack(src, rx0, ry0, rs, min_dist_px=max(rs, 256)) if t[2] >= 0.8]
    if len(twins) < 3:
        return x0, y0, _fallback(img, x0 - rx0, y0 - ry0, size), 0
    res = []
    for tx, ty, _ in twins:
        if tx < 0 or ty < 0 or tx + rs > W or ty + rs > H:
            continue
        tw = src.read_region(tx, ty, tx + rs, ty + rs).astype(np.float32) / 255.0
        sh = _shift(img, tw, max_shift=16.0)
        if sh is None:
            continue
        tw = ndi.shift(tw, sh, order=1, mode="nearest")
        if tw.std() > 1e-3:
            a, c = np.polyfit(tw.ravel()[::7], img.ravel()[::7], 1)
            tw = np.clip(a * tw + c, 0, 1)
        lo = ndi.grey_erosion(tw, size=(3, 3))
        hi = ndi.grey_dilation(tw, size=(3, 3))
        # near designed edges, stage placement and focus differ between twins by a pixel or two:
        # allow a share of the local edge contrast there (zero on flat field, where dust sits)
        rng5 = ndi.grey_dilation(tw, size=(5, 5)) - ndi.grey_erosion(tw, size=(5, 5))
        tol = 0.12 * rng5
        rr = np.maximum(0.0, np.maximum(lo - img, img - hi) - tol)
        edge = ndi.binary_dilation(rng5 > 0.05, iterations=3)
        # twins are not identical *designs* everywhere: test arrays vary junction size and labels
        # from cell to cell. Designed edges are the nets' job; the residual speaks only on flat
        # field (substrate, pad interiors), where the twin says nothing should be there at all
        # a true twin is the same design: it explains virtually the whole tile at full resolution.
        # Neighbouring test cells look alike at coarse scale but differ in wiring - reject them.
        # (after the edge mask: a twin that disagrees on more than a dust-sized patch of flat field
        # is a different design variant - a bigger junction pad, another label - not a reference)
        if (rr > 0.04).mean() > 0.03 or (np.where(edge, 0.0, rr) > 0.04).sum() > 600:
            continue
        res.append(np.where(edge, 0.0, rr))
    if len(res) < 3:
        return x0, y0, _fallback(img, x0 - rx0, y0 - ry0, size), len(res)
    r = np.median(np.stack(res), axis=0)
    # camera noise scales with brightness (bright pads are noisier): subtract a noise floor
    sigma = 1.4826 * float(np.median(np.abs(img - ndi.median_filter(img, size=3))))
    r = _drop_lines(np.maximum(0.0, r - 3.0 * sigma))
    ox, oy = x0 - rx0, y0 - ry0
    core = r[oy : oy + size, ox : ox + size]
    out[: core.shape[0], : core.shape[1]] = np.clip(np.rint(core * SCALE), 0, 255).astype(np.uint8)
    return x0, y0, out, len(res)


def _drop_lines(r: np.ndarray, thr: float = 0.03) -> np.ndarray:
    """Remove long thin residual streaks: an edge displaced along its whole length (stitching,
    a trace a pixel wider on this chip) is not dirt. Hairs are long too but the nets find those."""
    lab, n = ndi.label(r > thr)
    if not n:
        return r
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ln = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if ln > 30 and (lab[sl] == i).sum() < 3 * ln:
            r[sl][lab[sl] == i] = 0.0
    return r


def _fallback(img: np.ndarray, ox: int, oy: int, size: int) -> np.ndarray:
    """No trustworthy twins: flat field is its own reference (lone specks on bare substrate)."""
    out = np.zeros((size, size), np.uint8)
    # featureless or unique tile: bare substrate is its own reference (a smooth background),
    # which is exactly where lone dust specks sit; on unique structure this stays quiet
    # because the background model follows edges only at large scale
    # scored only away from designed edges: a speck is an isolated blob on a flat field
    sm = ndi.gaussian_filter(img, 2.0)
    grad = np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 1))
    edges = ndi.binary_dilation(grad > 0.05, iterations=5)
    bg = cv2.medianBlur(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8), 15).astype(np.float32) / 255.0
    r = np.zeros_like(img)  # only isolated blobs speak (fine meanders are texture, not edges)
    # keep blobs that sit on a flat field even if they produced their own small edge
    speck = ndi.binary_opening(np.abs(img - bg) > 0.04, structure=np.ones((2, 2)))
    lab, n = ndi.label(speck)
    if n:
        # fine structure (meanders, interdigitated fingers) blurs away at sigma 2: busy at full resolution too
        loc = ndi.uniform_filter(img, 9)
        texture = np.sqrt(np.maximum(0.0, ndi.uniform_filter(img * img, 9) - loc * loc))
        busy = (grad > 0.05) | (np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1)) > 0.12) | (texture > 0.025)
        ok = np.zeros_like(speck)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab[sl] == i
            if m.sum() > 400:
                continue
            y0s, x0s = max(0, sl[0].start - 10), max(0, sl[1].start - 10)
            y1s, x1s = sl[0].stop + 10, sl[1].stop + 10
            own = lab[y0s:y1s, x0s:x1s] == i
            ring = ndi.binary_dilation(own, iterations=8) & ~ndi.binary_dilation(own, iterations=3)
            # an isolated speck: its surroundings are featureless (not a bit of an edge or a letter)
            if ring.any() and busy[y0s:y1s, x0s:x1s][ring].mean() < 0.1:
                ok[y0s:y1s, x0s:x1s] |= ndi.binary_dilation(own, iterations=1)
        r = np.where(ok, np.abs(img - bg), r)
    r = _drop_lines(r)
    core = r[oy : oy + size, ox : ox + size]
    out[: core.shape[0], : core.shape[1]] = np.clip(np.rint(core * SCALE), 0, 255).astype(np.uint8)
    return out


_SRC: dict = {}


def reference_map(source_path: str | Path, out_path: str | Path, tile: int = 384, margin: int = 32, workers: int | None = None, progress=None, mask=None) -> dict:
    """Write residual map (zarr '0', uint8) and per-tile twin counts ('twins') to `out_path`."""
    import zarr

    from chipscan.io.readers import PyramidSource

    src = PyramidSource(source_path)
    H, W = src.shape
    g = zarr.open_group(str(out_path), mode="w")
    res = g.create_array("0", shape=(H, W), chunks=(1024, 1024), dtype="uint8", fill_value=0)
    ny, nx = -(-H // tile), -(-W // tile)
    tw = np.zeros((ny, nx), np.int16)
    jobs = [(str(source_path), x, y, tile, margin) for y in range(0, H, tile) for x in range(0, W, tile)]
    if mask is not None:  # off the chip: nothing to compare (twin count stays 0)
        jobs = [j for j in jobs if mask.any_in(j[1], j[2], j[1] + tile, j[2] + tile)]
    workers = workers or default_workers()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for x0, y0, arr, n in ex.map(_tile_residual, jobs, chunksize=16):
            h, w = min(tile, H - y0), min(tile, W - x0)
            res[y0 : y0 + h, x0 : x0 + w] = arr[:h, :w]
            tw[y0 // tile, x0 // tile] = n
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(jobs))
    g.create_array("twins", shape=tw.shape, dtype="int16")[:] = tw
    g.attrs.update({"tile": tile, "scale": SCALE})
    covered = float((tw >= 2).mean())
    return {"tiles": len(jobs), "covered_by_twins": round(covered, 3)}
