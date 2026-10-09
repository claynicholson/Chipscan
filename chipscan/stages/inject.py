"""Painters for self-supervised training on real tiles.

Each painter returns (image, mask) where mask marks what it changed visibly.  The speck and
trace detectors are trained on the *same* tiles with *all* painters active; only their own
category is labelled, everything else (the other category, text, designed marks) is painted
unlabelled.  That is what makes the two networks disagree in the right way: a round particle
lying on a trace is a speck, a straight-edged metal bridge across a gap is a trace error, and
letters, numbers and designed marks are neither.

Tones are always taken from the tile itself (dominant background, dark gap tone, bright metal
tone), so a painted defect looks like this microscope and this process, not like a cartoon.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def smooth_field(h: int, w: int, rng: np.random.Generator, scale: int = 8) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, tot, s = 1.0, 0.0, max(2, scale)
    while s >= 2:
        n = rng.uniform(size=(max(2, h // s), max(2, w // s))).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        s //= 2
    out /= max(tot, 1e-6)
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


class Tones:
    """Background / dark / bright tones of a tile, plus its structure masks."""

    def __init__(self, img: np.ndarray):
        hist, edges = np.histogram(img, bins=64, range=(0, 1))
        self.bg = float(edges[int(np.argmax(hist))] + 0.5 / 64)
        noise = float(1.4826 * np.median(np.abs(img - self.bg)[np.abs(img - self.bg) < 0.06])) if (np.abs(img - self.bg) < 0.06).any() else 0.01
        self.noise = max(noise, 0.004)
        t = max(0.035, 4 * self.noise)
        self.dark_mask = img < self.bg - t
        self.bright_mask = img > self.bg + t
        self.dark = float(np.median(img[self.dark_mask])) if self.dark_mask.sum() > 20 else max(0.0, self.bg - 0.15)
        self.bright = float(np.median(img[self.bright_mask])) if self.bright_mask.sum() > 20 else min(1.0, self.bg + 0.15)


def _blend(img: np.ndarray, mask: np.ndarray, tex: np.ndarray, soft: float) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.float32)
    a = cv2.GaussianBlur(m, (0, 0), soft) if soft > 0 else m
    a = np.clip(a * 1.25, 0, 1)
    out = img * (1 - a) + tex * a
    vis = (np.abs(out - img) > 0.02) & (mask > 0)
    return out.astype(np.float32), vis


# ----------------------------------------------------------------------------- specks (dirt)
def paint_speck(img: np.ndarray, rng: np.random.Generator, tones: Tones) -> tuple[np.ndarray, np.ndarray]:
    """Dust, particles, fibres, residue and probe scrub marks: irregular, own tone, soft edges."""
    h, w = img.shape
    kind = rng.choice(["dot", "particle", "particle", "fiber", "residue", "scrub"], p=[0.25, 0.3, 0.15, 0.1, 0.1, 0.1])
    m = np.zeros((h, w), np.uint8)
    cx, cy = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
    if kind == "scrub" and tones.bright_mask.sum() > 200 and rng.uniform() < 0.8:
        ys, xs = np.nonzero(tones.bright_mask)  # probe marks sit on pads
        k = int(rng.integers(len(xs)))
        cx, cy = float(xs[k]), float(ys[k])
    if kind == "dot":
        r = rng.uniform(0.7, 3.0)
        cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(round(r))), max(1, int(round(r * rng.uniform(0.6, 1))))), rng.uniform(0, 180), 0, 360, 1, -1)
    elif kind in ("particle", "residue"):
        r = rng.uniform(3, 14) if kind == "particle" else rng.uniform(8, 30)
        rr = int(np.ceil(r))
        x0, y0 = int(cx) - rr, int(cy) - rr
        f = smooth_field(2 * rr + 1, 2 * rr + 1, rng, scale=max(3, rr // 2))
        yy, xx = np.mgrid[-rr : rr + 1, -rr : rr + 1]
        blob = ((xx * xx + yy * yy) <= r * r) & (f > rng.uniform(0.3, 0.55))
        ys, xs = np.nonzero(blob)
        ys, xs = ys + y0, xs + x0
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        m[ys[ok], xs[ok]] = 1
    elif kind == "fiber":
        n = int(rng.integers(3, 7))
        pts = [np.array([cx, cy])]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(n):
            ang += rng.normal(0, 0.5)
            pts.append(pts[-1] + rng.uniform(6, 30) * np.array([np.cos(ang), np.sin(ang)]))
        cv2.polylines(m, [np.array(pts, np.int32)], False, 1, int(rng.integers(1, 4)), cv2.LINE_AA)
    else:  # scrub: a cluster of short parallel streaks
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        nrm = np.array([-d[1], d[0]])
        for _ in range(int(rng.integers(2, 7))):
            o = np.array([cx, cy]) + nrm * rng.normal(0, 3) + d * rng.normal(0, 3)
            ln = rng.uniform(5, 25)
            a, b = o - d * ln / 2, o + d * ln / 2
            cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, int(rng.integers(1, 3)))
    if m.sum() < 1:
        return img, np.zeros_like(img, bool)
    local = float(np.median(img[m > 0]))
    contrast = rng.uniform(0.03, 0.12) if rng.uniform() < 0.35 else rng.uniform(0.12, 0.45)  # many faint ones
    sign = -1 if rng.uniform() < 0.7 else 1  # particles mostly read dark in brightfield
    tone = float(np.clip(local + sign * contrast, 0, 1))
    if abs(tone - local) < 0.025:
        tone = float(np.clip(local - sign * contrast, 0, 1))
    tex = np.full_like(img, tone) + rng.normal(0, 0.015, img.shape).astype(np.float32)
    if kind in ("particle", "residue") and rng.uniform() < 0.4:  # bright core / dark rim (out of focus)
        core = cv2.erode(m, np.ones((3, 3), np.uint8))
        tex = np.where(core > 0, np.clip(local + 0.6 * contrast * -sign, 0, 1), tex).astype(np.float32)
    return _blend(img, m, tex, rng.uniform(0.4, 1.4))


# ----------------------------------------------------------------------------- trace defects
def _line_points(tones: Tones) -> tuple[np.ndarray, np.ndarray]:
    """Points on thin dark lines (gaps / trace edges) with their local direction."""
    dark = tones.dark_mask
    if dark.sum() < 40:
        return np.zeros((0, 2)), np.zeros((0, 2))
    lab, n = ndi.label(dark)
    sizes = ndi.sum(dark, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= 30)[0])
    if not keep.any():
        return np.zeros((0, 2)), np.zeros((0, 2))
    g = keep.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 3) for a in (gx * gx, gy * gy, gx * gy))
    ys, xs = np.nonzero(keep)
    # dominant gradient direction -> the line runs perpendicular to it
    th = 0.5 * np.arctan2(2 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs])
    coh = np.hypot(jxx[ys, xs] - jyy[ys, xs], 2 * jxy[ys, xs]) / (jxx[ys, xs] + jyy[ys, xs] + 1e-6)
    ok = coh > 0.6  # straight line, not a corner or a blob
    d = np.stack([-np.sin(th[ok]), np.cos(th[ok])], 1)
    return np.stack([xs[ok], ys[ok]], 1).astype(np.float32), d.astype(np.float32)


def _width_along(mask: np.ndarray, p: np.ndarray, n: np.ndarray, max_r: int = 25) -> tuple[int, int]:
    """Extent of `mask` through p along +-n: (back, forward) steps that stay inside."""
    h, w = mask.shape
    out = []
    for s in (-1, 1):
        k = 0
        while k < max_r:
            q = p + s * (k + 1) * n
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
                break
            k += 1
        out.append(k)
    return out[0], out[1]


def paint_trace_defect(img: np.ndarray, rng: np.random.Generator, tones: Tones) -> tuple[np.ndarray, np.ndarray]:
    """Lithography / etch defects that change the *shape* of traces with the process's own tones."""
    h, w = img.shape
    pts, dirs = _line_points(tones)
    if len(pts) == 0:
        return img, np.zeros_like(img, bool)
    k = int(rng.integers(len(pts)))
    p, d = pts[k], dirs[k]
    nrm = np.array([-d[1], d[0]], np.float32)
    back, fwd = _width_along(tones.dark_mask, p, nrm)
    lw = back + fwd + 1  # local dark-line width
    kind = rng.choice(["bridge", "cut", "bite", "protrusion", "width", "missing"], p=[0.22, 0.2, 0.16, 0.12, 0.15, 0.15])
    m = np.zeros((h, w), np.uint8)
    centre = p + nrm * (fwd - back) / 2.0
    if kind == "bridge":  # metal across the gap: a short
        ln = lw + 2 * rng.uniform(1, 4)
        wd = rng.uniform(2, 9)
        box = cv2.boxPoints((tuple(centre), (wd, ln), float(np.degrees(np.arctan2(d[1], d[0])))))
        cv2.fillPoly(m, [box.astype(np.int32)], 1)
        tone = tones.bg
    elif kind == "cut":  # dark band cutting through the conductor next to this line: an open
        side = 1 if rng.uniform() < 0.5 else -1
        reach = rng.uniform(5, 22)
        a = centre + side * nrm * (lw / 2 + reach / 2)
        box = cv2.boxPoints((tuple(a), (rng.uniform(2, 7), reach + lw), float(np.degrees(np.arctan2(d[1], d[0])))))
        cv2.fillPoly(m, [box.astype(np.int32)], 1)
        tone = tones.dark
    elif kind == "bite":  # notch from the gap into the metal
        side = 1 if rng.uniform() < 0.5 else -1
        r = rng.uniform(2, 6)
        c = centre + side * nrm * (lw / 2 + r * 0.5)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(max(1, r * rng.uniform(1, 2.5))), int(max(1, r))), float(np.degrees(np.arctan2(d[1], d[0]))), 0, 360, 1, -1)
        tone = tones.dark
    elif kind == "protrusion":  # metal bump into the gap (partial bridge)
        side = 1 if rng.uniform() < 0.5 else -1
        r = max(1.5, lw * rng.uniform(0.4, 0.8))
        c = centre + side * nrm * (lw / 2 - r * 0.4)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(max(1, r * rng.uniform(1, 3))), int(max(1, r))), float(np.degrees(np.arctan2(d[1], d[0]))), 0, 360, 1, -1)
        tone = tones.bg
    elif kind == "width":  # the line widens / narrows over a stretch
        L = rng.uniform(12, 70)
        grow = int(rng.integers(1, 3))
        seg = np.zeros((h, w), np.uint8)
        a, b = centre - d * L / 2, centre + d * L / 2
        cv2.line(seg, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, lw + 2 * grow + 4)
        line = tones.dark_mask.astype(np.uint8) & seg
        if rng.uniform() < 0.5:
            m = cv2.dilate(line, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) & seg & (~tones.dark_mask).astype(np.uint8)
            tone = tones.dark
        else:
            m = line & (1 - cv2.erode(line, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)))
            tone = tones.bg
    else:  # missing: a stretch of the line is gone
        L = rng.uniform(8, 60)
        seg = np.zeros((h, w), np.uint8)
        a, b = centre - d * L / 2, centre + d * L / 2
        cv2.line(seg, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, lw + 4)
        m = seg & cv2.dilate(tones.dark_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        tone = tones.bg
    if m.sum() < 2:
        return img, np.zeros_like(img, bool)
    tex = np.full_like(img, tone) + rng.normal(0, tones.noise, img.shape).astype(np.float32)
    return _blend(img, m, tex, rng.uniform(0.3, 0.8))


# ----------------------------------------------------------------------------- unlabelled designed content
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX)
_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789-.abcdefghijklmnopqrstuvwxyz"
_WORDS = ("um", "nm", "W5.0um", "L23.0um", "JJ", "N7", "W400nm", "L2000nm", "2.1um", "RETA-1", "AFT5", "A2", "x", "SPLIT", "TS")


def paint_text(img: np.ndarray, rng: np.random.Generator, tones: Tones) -> tuple[np.ndarray, np.ndarray]:
    """Chip labels and test-structure annotations ("L23.0um W5.0um"): designed, not defects."""
    h, w = img.shape
    if rng.uniform() < 0.5:
        txt = " ".join(rng.choice(_WORDS, size=int(rng.integers(1, 3))))
    else:
        txt = "".join(rng.choice(list(_CHARS), size=int(rng.integers(1, 9))))
    font = int(rng.choice(_FONTS))
    scale = float(rng.uniform(0.3, 1.3)) if rng.uniform() < 0.7 else float(rng.uniform(1.3, 4.0))
    thick = int(rng.integers(1, 2 + int(scale)))
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    layer = np.zeros((h, w), np.uint8)
    x = int(rng.uniform(-tw * 0.2, max(1.0, w - tw * 0.8)))
    y = int(rng.uniform(min(th, h - 1), h))
    cv2.putText(layer, txt, (x, y), font, scale, 1, thick, cv2.LINE_AA)
    if rng.uniform() < 0.3:
        layer = np.ascontiguousarray(np.rot90(layer, int(rng.integers(1, 4))))
    if rng.uniform() < 0.5:  # etched labels often read as outlines
        layer = layer & (1 - cv2.erode(layer, np.ones((3, 3), np.uint8)))
    if layer.sum() == 0:
        return img, np.zeros_like(img, bool)
    tone = tones.bright if rng.uniform() < 0.5 else tones.dark
    if abs(tone - tones.bg) < 0.05:
        tone = float(np.clip(tones.bg + rng.choice([-1, 1]) * 0.15, 0, 1))
    return _blend(img, layer, np.full_like(img, tone), 0.4)


def paint_mark(img: np.ndarray, rng: np.random.Generator, tones: Tones) -> tuple[np.ndarray, np.ndarray]:
    """Designed small features: crisp, axis-aligned rectangles, crosses, L-marks, pad pairs."""
    h, w = img.shape
    m = np.zeros((h, w), np.uint8)
    cx, cy = int(rng.uniform(8, w - 8)), int(rng.uniform(8, h - 8))
    kind = rng.choice(["rect", "cross", "ell", "dumbbell", "array"])
    a, b = int(rng.integers(2, 16)), int(rng.integers(2, 16))
    if kind == "rect":
        cv2.rectangle(m, (cx - a, cy - b), (cx + a, cy + b), 1, -1)
    elif kind == "cross":
        t = int(rng.integers(1, 4))
        cv2.rectangle(m, (cx - a - 4, cy - t), (cx + a + 4, cy + t), 1, -1)
        cv2.rectangle(m, (cx - t, cy - b - 4), (cx + t, cy + b + 4), 1, -1)
    elif kind == "ell":
        t = int(rng.integers(1, 4))
        cv2.rectangle(m, (cx, cy), (cx + a + 6, cy + t), 1, -1)
        cv2.rectangle(m, (cx, cy), (cx + t, cy + b + 6), 1, -1)
    elif kind == "dumbbell":  # a JJ-like pad pair with its link
        s = int(rng.integers(3, 8))
        gap = int(rng.integers(4, 14))
        cv2.rectangle(m, (cx - s, cy - gap - 2 * s), (cx + s, cy - gap), 1, -1)
        cv2.rectangle(m, (cx - s, cy + gap), (cx + s, cy + gap + 2 * s), 1, -1)
        cv2.rectangle(m, (cx - 1, cy - gap), (cx + 1, cy + gap), 1, -1)
        if rng.uniform() < 0.5:
            m = np.ascontiguousarray(m.T) if h == w else m
    else:  # a regular array of small squares
        s, p = int(rng.integers(2, 5)), int(rng.integers(8, 20))
        for i in range(int(rng.integers(2, 5))):
            for j in range(int(rng.integers(2, 5))):
                x, y = cx + i * p, cy + j * p
                cv2.rectangle(m, (x - s, y - s), (x + s, y + s), 1, -1)
    if m.sum() == 0:
        return img, np.zeros_like(img, bool)
    tone = tones.bright if rng.uniform() < 0.6 else tones.dark
    tex = np.full_like(img, tone)
    if rng.uniform() < 0.5:  # bright metal with a dark outline, as the real pads read
        edge = m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8)))
        tex = np.where(edge > 0, tones.dark, tex).astype(np.float32)
    return _blend(img, m, tex, 0.3)


PAINTERS = {"speck": paint_speck, "trace": paint_trace_defect, "text": paint_text, "mark": paint_mark}


def compose(img: np.ndarray, rng: np.random.Generator, counts: dict[str, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Paint `counts[name]` instances of each painter in random order; return image and per-kind masks.

    Later paint occludes earlier paint, so masks are trimmed where a later painter covered them.
    """
    tones = Tones(img)
    order = [k for k, n in counts.items() for _ in range(n)]
    rng.shuffle(order)
    masks = {k: np.zeros(img.shape, bool) for k in counts}
    out = img
    for k in order:
        out, vis = PAINTERS[k](out, rng, tones)
        for kk in masks:
            if kk != k:
                masks[kk] &= ~vis
        masks[k] |= vis
    return np.clip(out, 0, 1).astype(np.float32), masks
