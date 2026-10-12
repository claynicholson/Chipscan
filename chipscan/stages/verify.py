"""Reference check for candidates: is this pixel pattern explained by the scan's own repeats?

Every candidate (speck, trace error, component error) is compared with the same spot on its
look-alikes: other chips on the reticle, other cells of the same array, other rows of the same
meander.  Twins are found pitch-free by template matching a context tile along its row and
column (no assumption about chip pitch, which varies by tens of px with stage placement).

For each twin we build a tolerance envelope (min/max over a 3x3 neighbourhood, so a 1 px edge
shift is not a difference) and measure how far the candidate's own pixels fall outside it.  The
median over twins is the reference residual: a defect that only this location has stays large;
designed structure that every twin shares goes to zero; a defect on one twin does not matter.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource
from chipscan.segmentation.die2die import _lowres


def find_twins_slack(source: ImageSource, x: int, y: int, size: int, level: int = 3, min_ncc: float = 0.6, max_twins: int = 8, min_dist_px: int = 256, slack_px: int = 48) -> list[tuple[int, int, float]]:
    """Like die2die.find_twins, but the row (column) search is a band of +-slack_px: chips and
    cells are not placed on exact rows and columns (stage placement errors of tens of px)."""
    s = 2**level
    _, low = _lowres(source, level)
    h, w = low.shape
    tx0, ty0, ts = x // s, y // s, max(8, size // s)
    tile = low[ty0 : ty0 + ts, tx0 : tx0 + ts]
    if tile.shape != (ts, ts) or tile.std() < 1.0:
        return []
    sl = max(1, slack_px // s)
    out: list[tuple[int, int, float]] = []
    for axis in (0, 1):  # 0: along the row (x varies), 1: along the column (y varies)
        if axis == 0:
            a0, a1 = max(0, ty0 - sl), min(h, ty0 + ts + sl)
            band = low[a0:a1, :]
        else:
            a0, a1 = max(0, tx0 - sl), min(w, tx0 + ts + sl)
            band = low[:, a0:a1]
        if band.shape[0] < ts or band.shape[1] < ts:
            continue
        r = cv2.matchTemplate(band, tile, cv2.TM_CCOEFF_NORMED)
        prof = r.max(axis=0) if axis == 0 else r.max(axis=1)
        arg = r.argmax(axis=0) if axis == 0 else r.argmax(axis=1)
        here = tx0 if axis == 0 else ty0
        for i in np.argsort(prof)[::-1][:300]:
            if prof[i] < min_ncc:
                break
            if abs(int(i) - here) * s < min_dist_px:
                continue
            if axis == 0:
                cand = (int(i) * s, (a0 + int(arg[i])) * s, float(prof[i]))
            else:
                cand = ((a0 + int(arg[i])) * s, int(i) * s, float(prof[i]))
            if all(abs(cand[0] - c[0]) + abs(cand[1] - c[1]) >= min_dist_px for c in out):
                out.append(cand)
    out.sort(key=lambda t: -t[2])
    return out[:max_twins]


def _mask(d: dict, ox: int, oy: int, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    polys = [np.rint(np.asarray(p, np.float64) - [ox, oy]).astype(np.int32) for p in d.get("polygons_px") or [] if len(p) >= 3]
    if polys:
        cv2.fillPoly(m, polys, 1)
    if not m.any():
        b = d["bbox_px"]
        cv2.rectangle(m, (int(b["x0"] - ox), int(b["y0"] - oy)), (int(b["x1"] - ox), int(b["y1"] - oy)), 1, -1)
    return m.astype(bool)


def _shift(ref: np.ndarray, mov: np.ndarray, max_shift: float = 12.0) -> tuple[float, float] | None:
    from skimage.registration import phase_cross_correlation

    if ref.std() < 1e-3 or mov.std() < 1e-3:
        return (0.0, 0.0)
    try:
        s, _, _ = phase_cross_correlation(ref, mov, upsample_factor=4, normalization=None)
    except Exception:
        return None
    if np.hypot(*s) > max_shift:
        return None
    return float(s[0]), float(s[1])


def reference_residual(source: ImageSource, d: dict, context: int = 256, min_twins: int = 2) -> dict:
    """{'n_twins', 'residual' (median over twins of the mean out-of-envelope contrast inside the
    candidate), 'frac' (fraction of candidate pixels outside the envelope)}."""
    H, W = source.shape
    b = d["bbox_px"]
    cx, cy = 0.5 * (b["x0"] + b["x1"]), 0.5 * (b["y0"] + b["y1"])
    half = int(max(context // 2, 0.5 * max(b["x1"] - b["x0"], b["y1"] - b["y0"]) + 32))
    size = 2 * half
    if size > min(H, W):
        return {"n_twins": 0, "residual": None, "frac": None}
    # keep the window inside the image (candidates near the border sit off-centre in it)
    x0 = int(np.clip(int(cx) - half, 0, W - size))
    y0 = int(np.clip(int(cy) - half, 0, H - size))
    img = source.read_region(x0, y0, x0 + size, y0 + size).astype(np.float32) / 255.0
    own = _mask(d, x0, y0, img.shape)
    if not own.any():
        return {"n_twins": 0, "residual": None, "frac": None}
    twins = find_twins_slack(source, x0, y0, size, min_dist_px=max(size, 256))
    res_maps = []
    for tx, ty, _ in twins:
        if tx < 0 or ty < 0 or tx + size > W or ty + size > H:
            continue
        tw = source.read_region(tx, ty, tx + size, ty + size).astype(np.float32) / 255.0
        # align on everything except the candidate itself
        ref = np.where(own, np.median(img), img)
        sh = _shift(ref, tw)
        if sh is None:
            continue
        tw = ndi.shift(tw, sh, order=1, mode="nearest")
        # match brightness (exposure differs from frame to frame)
        a, c = np.polyfit(tw[~own].ravel(), img[~own].ravel(), 1) if (~own).sum() > 50 else (1.0, 0.0)
        tw = np.clip(a * tw + c, 0, 1)
        lo = ndi.grey_erosion(tw, size=(3, 3))
        hi = ndi.grey_dilation(tw, size=(3, 3))
        res_maps.append(np.maximum(0.0, np.maximum(lo - img, img - hi)))
    if len(res_maps) < min_twins:
        return {"n_twins": len(res_maps), "residual": None, "frac": None}
    r = np.median(np.stack(res_maps), axis=0)
    inside = r[own]
    return {"n_twins": len(res_maps), "residual": float(inside.mean()), "frac": float((inside > 0.04).mean())}


def verify(source: ImageSource, recs: list[dict], min_residual: float = 0.018, min_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split into (kept, explained). Candidates without enough twins are kept, marked unverified."""
    kept, explained = [], []
    for d in recs:
        r = reference_residual(source, d)
        d["n_twins"] = r["n_twins"]
        d["ref_residual"] = r["residual"]
        if r["residual"] is None:
            d["verified"] = False
            kept.append(d)
        elif r["residual"] < min_residual and r["frac"] < min_frac:
            explained.append(d)
        else:
            d["verified"] = True
            kept.append(d)
    return kept, explained
