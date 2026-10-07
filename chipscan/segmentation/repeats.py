"""Repeat suppression: a design-free second opinion for anomaly candidates.

Designed structure repeats (chips on a reticle, cells in an array, rows of
resonators); defects do not.  Given anomaly candidates, this module estimates
the dominant repetition pitches of the image from its autocorrelation, and for
each candidate compares its surroundings with the same location one pitch away
in each direction.  A candidate whose patch is reproduced elsewhere (high
normalized cross-correlation) is designed structure and is dropped; one that
differs from its repeats is kept, and gains a *repeat score* = how unlike its
twins it is.  No GDS, no registration.
"""

from __future__ import annotations

import numpy as np

from chipscan.io.readers import ImageSource


def estimate_pitches(source: ImageSource, max_level_px: int = 2500, min_pitch_frac: float = 0.03, max_pitch_frac: float = 0.6, n: int = 3) -> dict[str, list[float]]:
    """Dominant horizontal and vertical repetition pitches in level-0 px, by autocorrelation of a low-res level."""
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > max_level_px:
        level += 1
    s = 2**level
    img = source.read_level(level).astype(np.float32)
    img -= img.mean()
    out: dict[str, list[float]] = {}
    for axis, key in ((1, "x"), (0, "y")):
        prof_len = img.shape[axis]
        # 1-D autocorrelation of the image along the axis (mean over the other axis of the 2-D product)
        f = np.fft.rfft(img, axis=axis, n=2 * prof_len)
        ac = np.fft.irfft(f * np.conj(f), axis=axis)[: prof_len] if axis == 0 else np.fft.irfft(f * np.conj(f), axis=axis)[:, :prof_len]
        ac = ac.mean(axis=1 - axis)
        ac /= max(ac[0], 1e-9)
        lo, hi = int(prof_len * min_pitch_frac), int(prof_len * max_pitch_frac)
        seg = ac[lo:hi]
        peaks = [i + lo for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] > seg[i + 1] and seg[i] > 0.25]
        peaks.sort(key=lambda i: -ac[i])
        # keep distinct pitches (not multiples of a stronger one)
        kept: list[int] = []
        for p in peaks:
            if all(abs(p / q - round(p / q)) > 0.08 for q in kept):
                kept.append(p)
            if len(kept) >= n:
                break
        out[key] = [float(p * s) for p in kept]
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return 0.0 if d == 0 else float((a * b).sum() / d)


def _best_match(source: ImageSource, patch: np.ndarray, cx: int, cy: int, half: int, search: int) -> float:
    """Max NCC of `patch` against the neighbourhood around (cx, cy), allowing +/- `search` px of slack."""
    import cv2

    x0, y0 = cx - half - search, cy - half - search
    win = source.read_region(x0, y0, x0 + 2 * (half + search), y0 + 2 * (half + search))
    if win.shape[0] < patch.shape[0] + 2 or win.shape[1] < patch.shape[1] + 2:
        return 0.0
    res = cv2.matchTemplate(win.astype(np.float32), patch.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else 0.0


def suppress_repeats(
    source: ImageSource,
    candidates: list[dict],
    pitches: dict[str, list[float]] | None = None,
    half_px: int = 32,
    search_px: int = 6,
    match_ncc: float = 0.8,
    max_candidates: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split candidates into (kept, suppressed). Each record gains 'repeat_ncc' (best twin match)."""
    pitches = pitches or estimate_pitches(source)
    H, W = source.shape
    offsets: list[tuple[int, int]] = []
    for p in pitches.get("x", []):
        offsets += [(int(round(p)), 0), (-int(round(p)), 0), (2 * int(round(p)), 0), (-2 * int(round(p)), 0)]
    for p in pitches.get("y", []):
        offsets += [(0, int(round(p))), (0, -int(round(p))), (0, 2 * int(round(p))), (0, -2 * int(round(p)))]
    kept, dropped = [], []
    cands = candidates if max_candidates is None else sorted(candidates, key=lambda d: -d["score"] * d["area_px"])[:max_candidates]
    for d in cands:
        cx, cy = (int(round(v)) for v in d["centroid_px"])
        h = max(half_px, int(np.sqrt(d["area_px"])) + 8)
        if not offsets or cx - h < 0 or cy - h < 0 or cx + h > W or cy + h > H:
            d["repeat_ncc"] = None
            kept.append(d)
            continue
        patch = source.read_region(cx - h, cy - h, cx + h, cy + h)
        if patch.std() < 1.0:
            d["repeat_ncc"] = None
            kept.append(d)
            continue
        best = 0.0
        for dx, dy in offsets:
            tx, ty = cx + dx, cy + dy
            if tx - h - search_px < 0 or ty - h - search_px < 0 or tx + h + search_px > W or ty + h + search_px > H:
                continue
            best = max(best, _best_match(source, patch, tx, ty, h, search_px))
            if best >= match_ncc:
                break
        d["repeat_ncc"] = round(best, 3)
        (dropped if best >= match_ncc else kept).append(d)
    return kept, dropped
