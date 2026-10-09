"""Weight every finding by how likely it is to hurt the device.

What matters for a superconducting circuit is, in order: anything on or next to a Josephson
junction (shorts it, changes its critical current, adds lossy dielectric right where the field
is), then anything on a trace (can short a gap or open a line), then everything else (a speck
on bare ground plane is cosmetic).  Size scales all of it: a 5 um particle can bridge a 3 um
gap, a 0.5 um one cannot.

For each finding we measure, from the layout (layout net or GDS):
  d_jj    - distance from the finding's outline to the nearest component pixel (um)
  d_trace - distance to the nearest trace pixel (um)
  size    - equivalent diameter (um)
and combine them into a 0-100 risk with a level (critical / high / medium / low).  All the
ingredients are stored on the record so the number is explainable in the viewer.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

LEVELS = ((75, "critical"), (50, "high"), (25, "medium"), (0, "low"))
L_JJ_UM = 15.0  # interference length scale around a junction
L_TRACE_UM = 4.0  # around a trace / gap edge
SIZE_UM = 3.0  # particle size at which a speck is "large" (bridges typical gaps)


def level_of(score: float) -> str:
    for t, name in LEVELS:
        if score >= t:
            return name
    return "low"


def _finding_mask(d: dict, ox: int, oy: int, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    polys = [np.rint(np.asarray(p, np.float64) - [ox, oy]).astype(np.int32) for p in d.get("polygons_px") or [] if len(p) >= 3]
    if polys:
        cv2.fillPoly(m, polys, 1)
    if not m.any():
        b = d["bbox_px"]
        cv2.rectangle(m, (int(b["x0"] - ox), int(b["y0"] - oy)), (int(b["x1"] - ox), int(b["y1"] - oy)), 1, -1)
    return m.astype(bool)


def assess(findings: list[dict], class_map, pixel_size_um: float | None, window_um: float = 80.0) -> list[dict]:
    """Add d_jj_um, d_trace_um, size_um, on_jj, on_trace, risk (0-100) and risk_level to each finding."""
    px = pixel_size_um or 1.0
    H, W = class_map.shape
    R = int(np.ceil(window_um / px))
    for d in findings:
        b = d["bbox_px"]
        x0, y0 = int(max(0, b["x0"] - R)), int(max(0, b["y0"] - R))
        x1, y1 = int(min(W, b["x1"] + R)), int(min(H, b["y1"] + R))
        cls = np.asarray(class_map[y0:y1, x0:x1])
        own = _finding_mask(d, x0, y0, cls.shape)
        # the layout may have labelled the defect itself as structure: do not measure to yourself
        excl = ndi.binary_dilation(own, iterations=2) if d.get("group") != "component" else np.zeros_like(own)
        comp = (cls == 2) & ~excl
        trace = (cls == 1) & ~excl
        if d.get("group") == "component":
            d_jj = 0.0
        else:
            d_jj = float(ndi.distance_transform_edt(~comp)[own].min()) * px if comp.any() else window_um
        d_tr = float(ndi.distance_transform_edt(~trace)[own].min()) * px if trace.any() else window_um
        area_um2 = float(d.get("area_um2") or d.get("area_px", 1) * px * px)
        size = 2.0 * np.sqrt(area_um2 / np.pi)
        jj = float(np.exp(-d_jj / L_JJ_UM))
        tr = float(np.exp(-d_tr / L_TRACE_UM))
        sz = float(1.0 - np.exp(-size / SIZE_UM))
        g = d.get("group", "speck")
        conf = float(np.clip(d.get("score", 0.5), 0, 1))
        if g == "component":
            cls_name = d.get("class", "")
            base = 0.9 if cls_name == "component_missing" else 0.75
            r = base + 0.25 * min(1.0, (d.get("z") or 0) / 25.0)
        elif g == "trace":
            # a shape error is on a trace by construction; near a junction it matters most
            r = (0.45 + 0.35 * jj + 0.2 * sz) * (0.6 + 0.4 * conf)
        else:  # speck
            r = sz * (0.12 + 0.88 * max(jj, 0.65 * tr)) * (0.7 + 0.3 * conf)
        score = float(np.clip(100.0 * r, 0, 100))
        d.update(
            {
                "d_jj_um": round(d_jj, 2),
                "d_trace_um": round(d_tr, 2),
                "size_um": round(size, 2),
                "on_jj": d_jj <= px,
                "on_trace": d_tr <= px,
                "risk": round(score, 1),
                "risk_level": level_of(score),
                "risk_factors": {"jj": round(jj, 3), "trace": round(tr, 3), "size": round(sz, 3), "confidence": round(conf, 3)},
            }
        )
    return findings


def suppress_text(findings: list[dict], class_map, frac: float = 0.4, grow_px: int = 4) -> tuple[list[dict], list[dict]]:
    """Drop findings that lie mostly on text (labels, test-structure annotations)."""
    kept, dropped = [], []
    H, W = class_map.shape
    for d in findings:
        b = d["bbox_px"]
        x0, y0 = int(max(0, b["x0"] - grow_px)), int(max(0, b["y0"] - grow_px))
        x1, y1 = int(min(W, b["x1"] + grow_px)), int(min(H, b["y1"] + grow_px))
        cls = np.asarray(class_map[y0:y1, x0:x1])
        own = _finding_mask(d, x0, y0, cls.shape)
        text = ndi.binary_dilation(cls == 3, iterations=grow_px)
        on = float((text & own).sum()) / max(1, own.sum())
        (dropped if on >= frac else kept).append(d)
    return kept, dropped
