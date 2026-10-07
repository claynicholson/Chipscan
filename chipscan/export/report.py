"""JSON / CSV summaries and a downsampled overlay image of the result."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chipscan.detection.defects import Defect
from chipscan.geometry import WarpField
from chipscan.io.readers import ImageSource

_COLORS = {
    "short": (255, 0, 0),
    "open": (255, 0, 255),
    "particle": (255, 160, 0),
    "protrusion": (255, 255, 0),
    "notch": (0, 255, 255),
    "void": (0, 200, 255),
    "missing_feature": (200, 0, 255),
    "extra_feature": (255, 80, 80),
    "unclassified": (0, 255, 0),
}


def summary(defects: list[Defect], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    by_cls = Counter(d.cls for d in defects)
    by_kind = Counter(d.kind for d in defects)
    return {
        "n_defects": len(defects),
        "by_class": dict(sorted(by_cls.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "total_area_um2": float(sum(d.area_um2 for d in defects)),
        **(extra or {}),
    }


def write_json(path: str | Path, defects: list[Defect], extra: dict[str, Any] | None = None) -> None:
    Path(path).write_text(
        json.dumps({"summary": summary(defects, extra), "defects": [d.to_dict() for d in defects]}, indent=2, default=_json_default)
    )


def write_csv(path: str | Path, defects: list[Defect]) -> None:
    rows = [d.to_dict() for d in defects]
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def write_overlay(
    path: str | Path,
    source: ImageSource,
    defects: list[Defect],
    warp: WarpField,
    max_px: int = 4000,
    design_edges: np.ndarray | None = None,
    min_box_px: int = 6,
) -> None:
    """Downsampled image with defect boxes (colour by class). `design_edges` optional, same shape as the level used."""
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > max_px:
        level += 1
    s = 2**level
    img = source.read_level(level)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if design_edges is not None and design_edges.shape == img.shape:
        rgb[design_edges > 0.3] = (120, 255, 120)
    for d in defects:
        if d.bbox_px is None:
            continue
        x0, y0, x1, y1 = (int(v / s) for v in (d.bbox_px.x0, d.bbox_px.y0, d.bbox_px.x1, d.bbox_px.y1))
        if x1 - x0 < min_box_px:
            cx = (x0 + x1) // 2
            x0, x1 = cx - min_box_px // 2, cx + min_box_px // 2
        if y1 - y0 < min_box_px:
            cy = (y0 + y1) // 2
            y0, y1 = cy - min_box_px // 2, cy + min_box_px // 2
        r, g, b = _COLORS.get(d.cls, (0, 255, 0))
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (b, g, r), 1)
    cv2.imwrite(str(path), rgb)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
