"""Annotated images of a run: the scan with its findings outlined in their risk colours.

heatmap   the whole scan in grey under a colour wash of findings per mm², PNG

overview  the whole scan downscaled to `width` px, PNG
crop      a rectangle at full resolution (a coarser pyramid level if it would pass `max_px`), PNG
tiff      every pixel as a tiled, JPEG-compressed pyramid BigTIFF (QuPath, Fiji, PRECiV)

A finding big enough to see at the output scale is outlined along its polygon; a smaller one
(most specks, at any scale below full resolution) gets a ring around it, so it can be found.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chipscan.io.readers import ImageSource

LEVELS = ("low", "medium", "high", "critical")  # drawing order: the worst on top
COLORS = {"critical": (94, 77, 255), "high": (64, 159, 255), "medium": (78, 210, 242), "low": (150, 135, 125)}  # BGR, as in the viewer
BUCKET = 2048


class FindingIndex:
    """Findings bucketed by their level-0 bounding box: which ones touch a window."""

    def __init__(self, findings: list[dict]):
        self.findings = sorted(findings, key=lambda d: LEVELS.index(_level(d)))
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for i, d in enumerate(self.findings):
            b = d["bbox_px"]
            for by in range(int(b["y0"]) // BUCKET, int(b["y1"]) // BUCKET + 1):
                for bx in range(int(b["x0"]) // BUCKET, int(b["x1"]) // BUCKET + 1):
                    self.buckets.setdefault((by, bx), []).append(i)

    def query(self, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
        hit: set[int] = set()
        for by in range(int(y0) // BUCKET, int(y1) // BUCKET + 1):
            for bx in range(int(x0) // BUCKET, int(x1) // BUCKET + 1):
                hit.update(self.buckets.get((by, bx), ()))
        out = []
        for i in sorted(hit):
            b = self.findings[i]["bbox_px"]
            if b["x1"] >= x0 and b["x0"] <= x1 and b["y1"] >= y0 and b["y0"] <= y1:
                out.append(self.findings[i])
        return out


def _level(d: dict) -> str:
    lv = d.get("risk_level")
    return lv if lv in COLORS else "medium"


def select(findings: list[dict], include_low: bool = False, hidden_ids: set | None = None) -> list[dict]:
    hidden_ids = hidden_ids or set()
    return [d for d in findings if d.get("bbox_px") and d.get("id") not in hidden_ids and (include_low or _level(d) != "low")]


def draw(img: np.ndarray, findings: list[dict], x0: float, y0: float, s: float, ring_min: float = 6.0, line: int = 2) -> None:
    """Outline `findings` on `img` (BGR), whose pixel (0, 0) is level-0 (x0, y0) and whose pixels are s level-0 px."""
    for d in findings:
        col = COLORS[_level(d)]
        b = d["bbox_px"]
        ext = max(b["x1"] - b["x0"], b["y1"] - b["y0"]) / s
        polys = [np.asarray(q, np.float64) for q in d.get("polygons_px") or [] if len(q) >= 3]
        if ext >= 12 and polys:
            pts = [np.rint(((p - [x0, y0]) / s) * 16).astype(np.int32) for p in polys]
            cv2.polylines(img, pts, True, col, line, cv2.LINE_AA, shift=4)
        else:
            cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
            r = max(ring_min, ext / 2 + 4)
            cv2.circle(img, (int(round((cx - x0) / s * 16)), int(round((cy - y0) / s * 16))), int(round(r * 16)), col, line, cv2.LINE_AA, shift=4)


def with_legend(img: np.ndarray, title: str, counts: dict[str, int], um_per_px: float) -> np.ndarray:
    """A white strip above the image (so it hides no finding): title, risk key with counts, scale bar."""
    w = img.shape[1]
    f = max(0.8, w / 3500)  # text scale
    th = max(1, int(round(2 * f)))
    hb = int(70 * f)
    band = np.full((hb, w, 3), 255, np.uint8)
    y = int(46 * f)
    x = int(20 * f)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(band, title, (x, y), font, 0.9 * f, (30, 30, 30), th, cv2.LINE_AA)
    x += cv2.getTextSize(title, font, 0.9 * f, th)[0][0] + int(40 * f)
    for lv in reversed(LEVELS):
        if not counts.get(lv):
            continue
        cv2.circle(band, (x + int(10 * f), y - int(10 * f)), int(10 * f), COLORS[lv], th + 1, cv2.LINE_AA)
        t = f"{lv} {counts[lv]:,}"
        cv2.putText(band, t, (x + int(28 * f), y), font, 0.75 * f, (30, 30, 30), th, cv2.LINE_AA)
        x += int(28 * f) + cv2.getTextSize(t, font, 0.75 * f, th)[0][0] + int(30 * f)
    # scale bar at the right: a round length near 1/8 of the width
    target = w / 8 * um_per_px
    nice = max(v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if v <= max(1.0, target))
    L = int(round(nice / um_per_px))
    label = f"{nice / 1000:g} mm" if nice >= 1000 else f"{nice:g} um"
    tw = cv2.getTextSize(label, font, 0.75 * f, th)[0][0]
    xr = w - int(20 * f) - tw - int(12 * f) - L
    if xr > x:
        cv2.rectangle(band, (xr, y - int(14 * f)), (xr + L, y - int(4 * f)), (30, 30, 30), -1)
        cv2.putText(band, label, (xr + L + int(12 * f), y), font, 0.75 * f, (30, 30, 30), th, cv2.LINE_AA)
    return np.vstack([band, img])


def _counts(findings: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for d in findings:
        c[_level(d)] = c.get(_level(d), 0) + 1
    return c


def _bgr(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else np.ascontiguousarray(a[..., ::-1])


def overview(src: ImageSource, findings: list[dict], out: Path, width: int = 16000, title: str = "", px_um: float = 1.0) -> Path:
    H, W = src.shape
    width = min(width, W)
    k = 0
    while k + 1 < src.num_levels and W / 2 ** (k + 1) >= width:
        k += 1
    img = src.read_level(k)
    height = int(round(H * width / W))
    img = _bgr(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
    s = W / width
    draw(img, findings, 0, 0, s, ring_min=max(6.0, width / 1000), line=max(2, width // 4000))
    img = with_legend(img, title, _counts(findings), px_um * s)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return out


def crop(src: ImageSource, findings: list[dict], out: Path, bbox: tuple[int, int, int, int], max_px: float = 400e6, min_width: int = 1200, title: str = "", px_um: float = 1.0) -> Path:
    H, W = src.shape
    x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x1, y1 = min(W, int(bbox[2])), min(H, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the crop is outside the scan")
    k = 0
    while k + 1 < src.num_levels and (x1 - x0) * (y1 - y0) / 4**k > max_px:
        k += 1
    s = 2**k
    img = _bgr(src.read_region(x0, y0, x1, y1, level=k))
    if img.shape[1] < min_width:
        # a zoomed-in view: blow it up with crisp pixels, as the viewer shows it
        z = -(-min_width // img.shape[1])
        img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        s = s / z
    idx = FindingIndex(findings)
    sel = idx.query(x0 - 50 * s, y0 - 50 * s, x1 + 50 * s, y1 + 50 * s)
    draw(img, sel, x0, y0, s, ring_min=8.0, line=2)
    img = with_legend(img, title, _counts(sel), px_um * s)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return out


def tiff(src: ImageSource, findings: list[dict], out: Path, progress=None, tile: int = 512, quality: int = 85, px_um: float = 1.0) -> Path:
    """Pyramid BigTIFF: level 0 plus every coarser level in SubIFDs, annotated at each level's own scale."""
    import tifffile

    from chipscan import default_workers

    H, W = src.shape
    levels = [L for L in range(src.num_levels) if min(src.level_shape(L)) >= tile] or [0]
    idx = FindingIndex(findings)
    total = sum(-(-src.level_shape(L)[0] // tile) for L in levels)
    done = [0]

    def tiles(L: int):
        s = 2**L
        h, w = src.level_shape(L)
        for ty in range(0, h, tile):
            # one band of rows per step: each stored chunk is decoded once
            band = _bgr(src.read_region(0, ty * s, w * s, (ty + tile) * s, level=L))
            if band.shape[0] < tile or band.shape[1] < -(-w // tile) * tile:
                band = np.pad(band, ((0, tile - band.shape[0]), (0, -(-w // tile) * tile - band.shape[1]), (0, 0)))
            sel = idx.query(-50 * s, ty * s - 50 * s, W + 50 * s, (ty + tile) * s + 50 * s)
            draw(band, sel, 0, ty * s, s, ring_min=8.0 if L == 0 else 6.0, line=2)
            band = np.ascontiguousarray(band[..., ::-1])  # drawn in BGR (OpenCV), stored as RGB
            for tx in range(0, w, tile):
                yield band[:, tx : tx + tile]
            done[0] += 1
            if progress:
                progress(done[0], total)

    res = 1e4 / px_um  # px per cm
    with tifffile.TiffWriter(out, bigtiff=True) as tw:
        for n, L in enumerate(levels):
            h, w = src.level_shape(L)
            tw.write(
                tiles(L),
                shape=(h, w, 3),
                dtype="uint8",
                tile=(tile, tile),
                photometric="rgb",
                compression="jpeg",
                compressionargs={"level": quality},
                subifds=len(levels) - 1 if n == 0 else None,
                subfiletype=1 if n else 0,
                resolution=(res / 2**L, res / 2**L),
                resolutionunit="CENTIMETER",
                maxworkers=default_workers(),
            )
    return out


def density(findings: list[dict], shape: tuple[int, int], cell: float, sigma_cells: float = 1.0) -> np.ndarray:
    """Findings per cell (cell x cell level-0 px) at their centres, smoothed over ~sigma_cells cells."""
    H, W = shape
    gh, gw = -(-H // int(cell)), -(-W // int(cell))
    g = np.zeros((gh, gw), np.float32)
    for d in findings:
        b = d["bbox_px"]
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        g[min(gh - 1, max(0, int(cy // cell))), min(gw - 1, max(0, int(cx // cell)))] += 1
    if sigma_cells > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma_cells, borderType=cv2.BORDER_CONSTANT)
    return g


def heatmap(src: ImageSource, findings: list[dict], out: Path, width: int = 8000, cell_um: float = 500.0, title: str = "", px_um: float = 1.0) -> Path:
    """The scan in grey with a colour wash of how many findings there are per mm² (inferno: dark
    purple few, yellow most). The scale tops out at the 99th percentile so one dirty spot does
    not wash the rest out; nothing is drawn where there are none."""
    H, W = src.shape
    width = min(width, W)
    k = 0
    while k + 1 < src.num_levels and W / 2 ** (k + 1) >= width:
        k += 1
    height = int(round(H * width / W))
    s = W / width
    base = cv2.resize(src.read_level(k), (width, height), interpolation=cv2.INTER_AREA)
    if base.ndim == 3:
        base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    base = cv2.cvtColor((base.astype(np.float32) * 0.55 + 40).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # dimmed, so colour reads

    cell = max(1.0, cell_um / px_um)
    g = density(findings, (H, W), cell) / (cell_um / 1000) ** 2  # per mm²
    pos = g[g > 1e-3]
    top = float(np.percentile(pos, 99)) if pos.size else 1.0
    top = max(top, 1e-6)
    up = cv2.resize(g, (width, height), interpolation=cv2.INTER_CUBIC)
    t = np.clip(up / top, 0, 1)
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = np.clip(t * 4, 0, 0.85)[..., None]  # fades in from zero, keeps a hint of the chip under the worst
    img = (base * (1 - a) + col * a).astype(np.uint8)
    img = _heat_legend(img, title, len(findings), top, cell_um, px_um * s)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return out


def _heat_legend(img: np.ndarray, title: str, n: int, top: float, cell_um: float, um_per_px: float) -> np.ndarray:
    w = img.shape[1]
    f = max(0.8, w / 3500)
    th = max(1, int(round(2 * f)))
    hb = int(70 * f)
    band = np.full((hb, w, 3), 255, np.uint8)
    y = int(46 * f)
    x = int(20 * f)
    font = cv2.FONT_HERSHEY_SIMPLEX
    head = f"{title}   {n:,} findings" if title else f"{n:,} findings"
    cv2.putText(band, head, (x, y), font, 0.9 * f, (30, 30, 30), th, cv2.LINE_AA)
    x += cv2.getTextSize(head, font, 0.9 * f, th)[0][0] + int(40 * f)
    bw, bh = int(360 * f), int(18 * f)
    ramp = cv2.applyColorMap(np.linspace(0, 255, bw).astype(np.uint8)[None].repeat(bh, 0), cv2.COLORMAP_INFERNO)
    if x + bw + int(300 * f) < w:
        cv2.putText(band, "0", (x, y), font, 0.7 * f, (30, 30, 30), th, cv2.LINE_AA)
        x += int(24 * f)
        band[y - bh : y, x : x + bw] = ramp
        x += bw + int(10 * f)
        lab = f"{top:,.3g}+ per sq mm  (counted in {cell_um:g} um squares)"
        cv2.putText(band, lab, (x, y), font, 0.7 * f, (30, 30, 30), th, cv2.LINE_AA)
    return np.vstack([band, img])
