"""Defect gallery: one row per defect with image crop, design overlay, and detection.

Each row shows:  raw image | image + design edges (green) + detection outline (colour by class)
                 | predicted metal mask | difference mask (red = extra metal, magenta = missing)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chipscan.design import Design
from chipscan.detection.defects import Defect
from chipscan.export.report import _COLORS
from chipscan.geometry import BBox, WarpField
from chipscan.io.readers import ImageSource


def _edges(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0


def defect_gallery(
    source: ImageSource,
    design: Design,
    warp: WarpField,
    defects: list[Defect],
    out_path: str | Path,
    pred: np.ndarray | None = None,
    n: int = 24,
    crop_px: int = 96,
    cell_px: int = 192,
    cols: int = 2,
    truth: list[dict] | None = None,
) -> Path:
    """Write a PNG montage. `pred` is the full-size predicted metal probability (uint8 0..255) or None."""
    defects = sorted(defects, key=lambda d: (-_sev(d.cls), -d.area_um2))[:n]
    panels = []
    for d in defects:
        if d.bbox_px is None:
            continue
        cx, cy = d.bbox_px.center
        half = max(crop_px // 2, int(max(d.bbox_px.width, d.bbox_px.height) * 0.75))
        x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
        img = source.read_region(x0, y0, x1, y1, level=0)
        bb = BBox(x0, y0, x1, y1)
        des = design.render(warp.local_affine(bb), bb)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        over = rgb.copy()
        over[_edges(des)] = (90, 230, 90)
        r, g, b = _COLORS.get(d.cls, (0, 255, 0))
        for poly in d.polygons_um:
            p = warp.world_to_px(poly) - [x0, y0]
            cv2.polylines(over, [np.rint(p).astype(np.int32)], True, (b, g, r), 1)
        if truth:
            for t in truth:
                tp = warp.world_to_px(np.array([[t["x_um"], t["y_um"]]]))[0] - [x0, y0]
                if 0 <= tp[0] < x1 - x0 and 0 <= tp[1] < y1 - y0:
                    cv2.drawMarker(over, (int(tp[0]), int(tp[1])), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
        if pred is not None:
            pm = _read(pred, x0, y0, x1, y1)
            predb = pm > 127
            diff = np.zeros_like(rgb)
            diff[predb & (des == 0)] = (0, 0, 255)
            diff[(~predb) & (des > 0)] = (255, 0, 255)
            pm_rgb = cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR)
        else:
            pm_rgb = np.zeros_like(rgb)
            diff = np.zeros_like(rgb)
        row = [rgb, over, pm_rgb, diff]
        row = [cv2.resize(a, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST) for a in row]
        strip = np.hstack(row)
        label = f"#{d.id} {d.cls} ({d.kind.replace('_metal', '')}) {d.area_um2:.0f} um2  score {d.score:.2f}"
        bar = np.zeros((22, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, strip]))
    if not panels:
        panels = [np.zeros((64, cell_px * 4, 3), np.uint8)]
        cv2.putText(panels[0], "no defects", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    hdr = np.zeros((26, panels[0].shape[1], 3), np.uint8)
    for i, t in enumerate(("image", "design edges + detection", "predicted metal", "diff: red extra / magenta missing")):
        cv2.putText(hdr, t, (4 + i * cell_px, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    rows = []
    for i in range(0, len(panels), cols):
        grp = panels[i : i + cols]
        while len(grp) < cols:
            grp.append(np.zeros_like(panels[0]))
        rows.append(np.hstack([np.hstack([p, np.full((p.shape[0], 8, 3), 40, np.uint8)]) for p in grp]))
    hdr_row = np.hstack([np.hstack([hdr, np.full((hdr.shape[0], 8, 3), 40, np.uint8)])] * cols)
    out = np.vstack([hdr_row, *rows])
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), out)
    return out_path


def _read(arr, x0, y0, x1, y1) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.asarray(arr[cy0:cy1, cx0:cx1])
    return out


_SEV = {"short": 5, "open": 5, "missing_feature": 4, "extra_feature": 4, "particle": 3, "protrusion": 2, "notch": 2, "void": 1}


def _sev(c: str) -> int:
    return _SEV.get(c, 0)
