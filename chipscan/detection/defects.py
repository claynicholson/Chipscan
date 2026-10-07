"""Defect records and a rule-based taxonomy.

A defect is a connected blob where the segmented image disagrees with the design,
outside the tolerance band around design edges.  Two polarities:

* ``extra_metal``   : metal where the design has none.
* ``missing_metal`` : no metal where the design has some.

:class:`RuleClassifier` turns polarity + topology into physically meaningful
classes.  It is deliberately simple and overridable: subclass or pass your own
``classify(defect, context)`` for product-specific rules (airbridges, junctions...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
from scipy import ndimage as ndi

from chipscan.geometry import BBox

EXTRA = "extra_metal"
MISSING = "missing_metal"


@dataclass
class Defect:
    id: int
    kind: str  # EXTRA or MISSING
    cls: str  # taxonomy class, e.g. short/open/particle/void/protrusion/notch
    bbox_um: BBox
    centroid_um: tuple[float, float]
    area_um2: float
    area_px: int
    score: float  # mean |prob - design| inside the blob (0..1)
    polygons_um: list[np.ndarray] = field(default_factory=list)  # outer contours in um
    bbox_px: BBox | None = None
    tile_index: int = -1
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "class": self.cls,
            "bbox_um": self.bbox_um.to_dict(),
            "bbox_px": None if self.bbox_px is None else self.bbox_px.to_dict(),
            "centroid_um": list(self.centroid_um),
            "area_um2": self.area_um2,
            "area_px": self.area_px,
            "score": self.score,
            "n_polygons": len(self.polygons_um),
            "tile": self.tile_index,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


@dataclass
class BlobContext:
    """What a classifier may look at: the blob and the design around it (tile-local px)."""

    blob: np.ndarray  # bool (h,w)
    design: np.ndarray  # bool (h,w) metal mask
    pred: np.ndarray  # bool (h,w) predicted metal
    pixel_size_um: float


class Classifier(Protocol):
    def classify(self, kind: str, ctx: BlobContext) -> tuple[str, dict[str, Any]]: ...


class RuleClassifier:
    """Topology rules.

    extra_metal:  touches >=2 design metal components -> ``short``;
                  touches 1 -> ``protrusion``; touches 0 -> ``particle``.
    missing_metal: removing it disconnects its design component -> ``open``;
                  touches the component's edge -> ``notch``; interior -> ``void``.
    Very large blobs of either kind become ``missing_feature`` / ``extra_feature``.
    """

    def __init__(self, large_um2: float = 2500.0, touch_px: int = 1, tolerance_px: int = 0):
        self.large_um2 = large_um2
        # blobs were carved out of the tolerance band around design edges, so to test
        # what they touch we must grow them back across that band
        self.touch_px = touch_px + tolerance_px + 1

    def classify(self, kind: str, ctx: BlobContext) -> tuple[str, dict[str, Any]]:
        blob = ctx.blob
        area_um2 = float(blob.sum()) * ctx.pixel_size_um**2
        grown = ndi.binary_dilation(blob, iterations=self.touch_px)
        meta: dict[str, Any] = {}
        if kind == EXTRA:
            lab, n = ndi.label(ctx.design)
            touched = set(np.unique(lab[grown & (lab > 0)]).tolist())
            meta["touched_components"] = len(touched)
            if area_um2 > self.large_um2:
                return "extra_feature", meta
            if len(touched) >= 2:
                return "short", meta
            if len(touched) == 1:
                return "protrusion", meta
            return "particle", meta
        # MISSING
        lab, n = ndi.label(ctx.design)
        ids = np.unique(lab[blob & (lab > 0)])
        if area_um2 > self.large_um2:
            return "missing_feature", meta
        if len(ids) == 0:
            return "void", meta
        comp = lab == ids[0]
        _, n_before = ndi.label(comp)
        _, n_after = ndi.label(comp & ~grown)
        meta["splits_component"] = bool(n_after > n_before)
        if n_after > n_before:
            return "open", meta
        edge = comp ^ ndi.binary_erosion(comp, iterations=1, border_value=0)
        if (grown & edge).any() and (grown & ~comp).any():
            return "notch", meta
        return "void", meta


def blobs_from_mask(diff: np.ndarray, min_area_px: int) -> list[tuple[np.ndarray, BBox, int]]:
    """Connected components of a binary difference mask -> [(component_bool, bbox_px_local, area)]."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            continue
        out.append((lab == i, BBox(float(x), float(y), float(x + w), float(y + h)), int(area)))
    return out


def contours_px(blob: np.ndarray) -> list[np.ndarray]:
    cs, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            out.append(pts + 0.5)  # pixel centres
    return out
