"""Stitching ghosts: overlap bands where two misregistered camera frames were blended.

PRECiV blends neighbouring frames across their overlap.  When the frames are registered a few
px apart, every structure in the blend band appears twice, offset by the misregistration
(often ~5-15 px) - a JJ there looks "deformed" and dirt-like ghost edges appear.  The blend is
not invertible from the exported mosaic alone (each pixel mixes two unknowns), so the honest
handling is to find these bands and keep their findings out of the defect list; the real fix
is re-stitching from the raw frames.

Detection: a ghost doubles every edge at a fixed lag.  Along each seam (horizontal seams give
vertical doubling, vertical seams horizontal doubling) the edge-autocorrelation at lags 6-30 px
is computed in short windows; windows that stand well above the local baseline, with a
consistent lag, form a ghost zone.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource

LAGS = np.arange(7, 31)  # 6 px and below is edge width / dense designed pitch, not a ghost


def _doubling_all(img: np.ndarray) -> np.ndarray:
    """Normalised edge autocorrelation for every lag in LAGS (doubling along axis 0)."""
    g = np.abs(ndi.sobel(ndi.gaussian_filter(img, 1.0), axis=0))
    p = g - g.mean(1, keepdims=True)
    out = np.empty(len(LAGS), np.float32)
    for k, L in enumerate(LAGS):
        a, b = p[:-L], p[L:]
        out[k] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
    return out


def _doubling(img: np.ndarray, axis: int) -> tuple[float, int]:
    """Max normalised edge autocorrelation over LAGS along `axis` (0: vertical doubling)."""
    g = np.abs(ndi.sobel(ndi.gaussian_filter(img, 1.0), axis=axis))
    p = g - (g.mean(1, keepdims=True) if axis == 0 else g.mean(0, keepdims=True))
    best, lag = -1.0, 0
    for L in LAGS:
        a, b = (p[:-L], p[L:]) if axis == 0 else (p[:, :-L], p[:, L:])
        v = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
        if v > best:
            best, lag = v, int(L)
    return best, lag


def find_ghosts(
    source: ImageSource,
    frame_px: tuple[float, float],
    phase_px: tuple[float, float] | None = None,
    reach: int = 200,
    window: int = 80,
    step: int = 20,
    segment: int = 640,
    min_excess: float = 0.12,
    progress=None,
) -> list[dict]:
    """Ghost zones [{'x0','y0','x1','y1','lag','score','axis'}] along every frame seam."""
    H, W = source.shape
    Px, Py = frame_px
    if phase_px is None:
        phase_px = seam_phase_from_shading(source, frame_px)
    ox, oy = phase_px
    zones: list[dict] = []
    jobs = []
    for ky in range(1, int(H / Py) + 1):  # horizontal seams -> vertical doubling
        ys = oy + ky * Py
        if reach < ys < H - reach:
            for x0 in range(0, W - segment + 1, segment):
                jobs.append(("h", ys, x0))
    for kx in range(1, int(W / Px) + 1):  # vertical seams -> horizontal doubling
        xs = ox + kx * Px
        if reach < xs < W - reach:
            for y0 in range(0, H - segment + 1, segment):
                jobs.append(("v", xs, y0))
    for j, (kind, c, a0) in enumerate(jobs):
        c = int(round(c))
        if kind == "h":
            strip = source.read_region(a0, c - reach, a0 + segment, c + reach).astype(np.float32)
        else:
            strip = source.read_region(c - reach, a0, c + reach, a0 + segment).astype(np.float32).T
        if strip.std() < 3:
            continue
        pos = list(range(0, 2 * reach - window + 1, step))
        sc = []
        for p0 in pos:
            sc.append(_doubling(strip[p0 : p0 + window], axis=0))
        vals = np.array([s for s, _ in sc])
        base = float(np.median(vals))
        hot = vals > base + min_excess
        if hot.sum() < 2:
            continue
        lags = np.array([lg for _, lg in sc])
        hl = lags[hot]
        lag = int(np.median(hl))
        if np.mean(np.abs(hl - lag) <= 2) < 0.6:  # the doubling must have one consistent offset
            continue
        idx = np.nonzero(hot)[0]
        b0 = pos[idx.min()]
        b1 = pos[idx.max()] + window
        # a blend band straddles its seam; doubling far from the seam is designed repetition
        if not (b0 - 40 <= reach <= b1 + 40) or b1 - b0 > 2 * reach - 40:
            continue
        z = {"lag": lag, "score": round(float(vals.max() - base), 3), "axis": "y" if kind == "h" else "x"}
        if kind == "h":
            z.update({"x0": a0, "x1": a0 + segment, "y0": c - reach + b0, "y1": c - reach + b1})
        else:
            z.update({"x0": c - reach + b0, "x1": c - reach + b1, "y0": a0, "y1": a0 + segment})
        zones.append(z)
        if progress:
            progress(j + 1, len(jobs))
    return zones


def seam_phase_from_shading(source: ImageSource, frame_px: tuple[float, float]) -> tuple[float, float]:
    """Seam offsets from the per-frame shading: the fold of the background has its step at the seam."""
    import warnings

    from chipscan.stages.straighten import _background

    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > 13000:
        level += 1
    f = 2**level
    img = source.read_level(level).astype(np.float32)
    ratio, _ = _background(img, f)
    out = []
    for axis, P in ((0, frame_px[0]), (1, frame_px[1])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(ratio, axis=axis)
        x = np.arange(len(prof)) * f
        ok = ~np.isnan(prof)
        nb = max(8, int(P / f))
        b = ((x % P) / P * nb).astype(int) % nb
        m = np.bincount(b[ok], prof[ok], nb) / np.maximum(np.bincount(b[ok], None, nb), 1)
        d = np.abs(np.diff(np.r_[m, m[0]]))
        out.append(float((int(np.argmax(d)) + 1) * P / nb))
    return out[0], out[1]


def estimate_seam_phase(source: ImageSource, frame_px: tuple[float, float]) -> tuple[float, float]:
    """Where the seams fall: the offset at which edge doubling is strongest, folded on the frame size."""
    H, W = source.shape
    Px, Py = frame_px
    rng = np.random.default_rng(0)
    acc_y = np.zeros(int(Py) // 20 + 1)
    acc_x = np.zeros(int(Px) // 20 + 1)
    for _ in range(60):
        x0 = int(rng.uniform(0, W - 1300))
        y0 = int(rng.uniform(0, H - 1000))
        blk = source.read_region(x0, y0, x0 + 1280, y0 + 1000).astype(np.float32)
        if blk.std() < 3:
            continue
        for r in range(0, 1000 - 80, 20):
            s, _ = _doubling(blk[r : r + 80, :640], axis=0)
            acc_y[int(((y0 + r + 40) % Py) / 20)] += s
        for c in range(0, 1280 - 80, 20):
            s, _ = _doubling(blk[:640, c : c + 80].T, axis=0)
            acc_x[int(((x0 + c + 40) % Px) / 20)] += s
    return float(np.argmax(acc_x) * 20 + 10), float(np.argmax(acc_y) * 20 + 10)


def mark_findings(findings: list[dict], zones: list[dict], margin: int = 10) -> int:
    """Flag findings whose centre lies in a ghost zone; returns how many."""
    n = 0
    for d in findings:
        cx, cy = d["centroid_px"]
        for z in zones:
            if z["x0"] - margin <= cx <= z["x1"] + margin and z["y0"] - margin <= cy <= z["y1"] + margin:
                d["in_stitch_ghost"] = True
                n += 1
                break
    return n


def scan_ghosts(
    source: ImageSource,
    frame_px: tuple[float, float],
    window: int = 80,
    step: int = 20,
    segment: int = 640,
    min_excess: float = 0.12,
    max_band: int = 240,
    grid_tol: float = 80.0,
    progress=None,
) -> tuple[list[dict], dict]:
    """Seam-agnostic ghost search over the whole scan.

    1. every 640 px strip is scored for edge doubling in short windows along the strip;
    2. short runs well above the strip's running baseline, with one consistent lag, are
       candidate blend bands;
    3. real blend bands sit on the camera-frame grid: the grid phase is the mode of the
       candidates' positions modulo the frame size, and only candidates on it are kept.
    """
    H, W = source.shape
    Px, Py = frame_px
    cands: list[dict] = []
    strips = [("h", x0) for x0 in range(0, W - segment + 1, segment)] + [("v", y0) for y0 in range(0, H - segment + 1, segment)]
    for j, (kind, a0) in enumerate(strips):
        L = H if kind == "h" else W
        pos, prof = [], []
        for c0 in range(0, L - window, 2048):
            c1 = min(L, c0 + 2048 + window)
            blk = source.read_region(a0, c0, a0 + segment, c1) if kind == "h" else source.read_region(c0, a0, c1, a0 + segment).T
            blk = blk.astype(np.float32)
            for r in range(0, blk.shape[0] - window + 1, step):
                if c0 + r >= c0 + 2048 and c1 < L:
                    break
                w = blk[r : r + window]
                prof.append(_doubling_all(w) if w.std() >= 3 else np.zeros(len(LAGS), np.float32))
                pos.append(c0 + r + window // 2)
        if len(prof) < 10:
            continue
        prof = np.stack(prof)  # (n_windows, n_lags)
        # each lag against its own running baseline: designed repetition is steady along the strip,
        # a blend band lifts one lag in a short stretch
        base = ndi.median_filter(prof, size=(41, 1), mode="nearest")
        exc = prof - base
        best_lag = exc.argmax(1)
        best = exc.max(1)
        hot = best > min_excess
        lab, n = ndi.label(hot)
        for i in range(1, n + 1):
            idx = np.nonzero(lab == i)[0]
            if len(idx) < 2 or (idx[-1] - idx[0]) * step + window > max_band:
                continue
            lags = LAGS[best_lag[idx]]
            lag = int(np.median(lags))
            if np.mean(np.abs(lags - lag) <= 2) < 0.6:
                continue
            c = float(np.mean([pos[k] for k in idx]))
            half = (idx[-1] - idx[0]) * step / 2 + window / 2
            if c < 300 or c > L - 300:  # scan edges: cropped frames and padding, not blends
                continue
            cands.append({"kind": kind, "a0": a0, "c": c, "half": half, "lag": lag, "score": float(best[idx].max())})
        if progress:
            progress(j + 1, len(strips))
    zones: list[dict] = []
    phases = {}
    for kind, P in (("h", Py), ("v", Px)):
        cs = [c for c in cands if c["kind"] == kind]
        if len(cs) < 3:
            continue
        m = np.array([c["c"] % P for c in cs])
        hist, edges = np.histogram(m, bins=int(P // 20), range=(0, P), weights=[c["score"] for c in cs])
        ph = float(edges[int(np.argmax(hist))] + 10)
        phases[kind] = ph
        for c in cs:
            dist = abs(((c["c"] - ph) + P / 2) % P - P / 2)
            if dist > grid_tol:
                continue
            lo, hi = int(c["c"] - c["half"]), int(c["c"] + c["half"])
            if kind == "h":
                zones.append({"x0": c["a0"], "x1": c["a0"] + segment, "y0": lo, "y1": hi, "lag": c["lag"], "score": round(c["score"], 3), "axis": "y"})
            else:
                zones.append({"x0": lo, "x1": hi, "y0": c["a0"], "y1": c["a0"] + segment, "lag": c["lag"], "score": round(c["score"], 3), "axis": "x"})
    return zones, {"n_candidates": len(cands), "phase": phases}


def ghost_at(source: ImageSource, x: float, y: float, reach: int = 400, width: int = 640, window: int = 80, step: int = 20, min_excess: float = 0.12, min_edge: float = 3.0) -> dict | None:
    """Is (x, y) inside a stitching blend band? Checks both orientations around the point.

    Returns {'axis', 'lag', 'score', 'band': (lo, hi)} or None.  Windows with too little edge
    content are ignored (the autocorrelation of near-empty substrate is just noise)."""
    H, W = source.shape
    best = None
    for kind in ("h", "v"):
        if kind == "h":
            x0 = int(np.clip(x - width // 2, 0, max(0, W - width)))
            c0 = int(np.clip(y - reach, 0, max(0, H - 2 * reach)))
            blk = source.read_region(x0, c0, x0 + width, c0 + 2 * reach).astype(np.float32)
            here = y - c0
        else:
            y0 = int(np.clip(y - width // 2, 0, max(0, H - width)))
            c0 = int(np.clip(x - reach, 0, max(0, W - 2 * reach)))
            blk = source.read_region(c0, y0, c0 + 2 * reach, y0 + width).astype(np.float32).T
            here = x - c0
        pos, prof, ok = [], [], []
        for r in range(0, blk.shape[0] - window + 1, step):
            w = blk[r : r + window]
            g = np.abs(ndi.sobel(ndi.gaussian_filter(w, 1.0), axis=0))
            ok.append(g.mean() >= min_edge)
            prof.append(_doubling_all(w) if ok[-1] else np.zeros(len(LAGS), np.float32))
            pos.append(r + window // 2)
        prof, ok, pos = np.stack(prof), np.array(ok), np.array(pos)
        if ok.sum() < 8:
            continue
        base = np.median(prof[ok], axis=0)  # this strip's normal doubling at every lag
        exc = np.where(ok[:, None], prof - base[None, :], 0.0)
        bl = exc.argmax(1)
        hot = exc.max(1) > min_excess
        lab, n = ndi.label(hot)
        k = int(np.argmin(np.abs(pos - here)))
        # the point's own window, or its neighbour, must be in a hot run
        cand = [lab[k]] + ([lab[k - 1]] if k > 0 else []) + ([lab[k + 1]] if k + 1 < len(lab) else [])
        for li in [c for c in cand if c > 0]:
            idx = np.nonzero(lab == li)[0]
            if len(idx) < 2 or (idx[-1] - idx[0]) * step + window > 260:
                continue
            lags = LAGS[bl[idx]]
            lag = int(np.median(lags))
            if np.mean(np.abs(lags - lag) <= 2) < 0.6:
                continue
            sc = float(exc[idx].max())
            if best is None or sc > best["score"]:
                best = {"axis": "y" if kind == "h" else "x", "lag": lag, "score": round(sc, 3), "band": (int(c0 + pos[idx[0]] - window // 2), int(c0 + pos[idx[-1]] + window // 2))}
    return best


class _Masked:
    """read_region with a rectangle replaced by its surroundings' median (to hide the finding)."""

    def __init__(self, src: ImageSource, box: tuple[int, int, int, int]):
        self.src, self.box, self.shape = src, box, src.shape

    def read_region(self, x0, y0, x1, y1, level=0):
        a = self.src.read_region(x0, y0, x1, y1).copy()
        bx0, by0, bx1, by1 = self.box
        ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        if ix1 > ix0 and iy1 > iy0:
            a[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = np.median(a)
        return a


def finding_in_ghost(source: ImageSource, d: dict, side: int = 420, pad: int = 30) -> dict | None:
    """A finding is a stitching ghost when the same blend band (same orientation, similar lag,
    overlapping rows/cols) is found on both sides of it with the finding itself masked out."""
    b = d["bbox_px"]
    box = (int(b["x0"]) - pad, int(b["y0"]) - pad, int(b["x1"]) + pad, int(b["y1"]) + pad)
    src = _Masked(source, box)
    cx, cy = d["centroid_px"]
    out = None
    for axis, offs in (("y", ((-side, 0), (side, 0))), ("x", ((0, -side), (0, side)))):
        res = []
        for dx, dy in offs:
            g = ghost_at(src, cx + dx, cy + dy, width=480)
            res.append(g if g and g["axis"] == axis else None)
        if all(res):
            g1, g2 = res
            lo, hi = max(g1["band"][0], g2["band"][0]), min(g1["band"][1], g2["band"][1])
            pos = cy if axis == "y" else cx
            if abs(g1["lag"] - g2["lag"]) <= 3 and hi > lo and lo - 30 <= pos <= hi + 30:
                out = {"axis": axis, "lag": int(round((g1["lag"] + g2["lag"]) / 2)), "score": min(g1["score"], g2["score"]), "band": (lo, hi)}
    return out
