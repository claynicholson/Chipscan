"""Die-to-database comparison: predicted metal vs. rendered design -> defects."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from chipscan.config import DetectionConfig
from chipscan.detection.defects import (
    EXTRA,
    MISSING,
    BlobContext,
    Classifier,
    Defect,
    RuleClassifier,
    blobs_from_mask,
    contours_px,
)
from chipscan.geometry import BBox, WarpField


def tolerance_band(design: np.ndarray, tolerance_px: int) -> np.ndarray:
    """Pixels within `tolerance_px` of a design edge (ignored during comparison)."""
    d = design.astype(bool)
    edge = d ^ ndi.binary_erosion(d, border_value=0)
    edge |= d ^ ndi.binary_dilation(d, border_value=0)
    if tolerance_px <= 0:
        return edge
    return ndi.binary_dilation(edge, iterations=tolerance_px)


def diff_masks(prob: np.ndarray, design: np.ndarray, threshold: float, tolerance_px: int) -> tuple[np.ndarray, np.ndarray]:
    pred = prob > threshold
    d = design.astype(bool)
    band = tolerance_band(d, tolerance_px)
    extra = pred & ~d & ~band
    missing = d & ~pred & ~band
    return extra, missing


def compare_tile(
    prob: np.ndarray,
    design: np.ndarray,
    core_origin_px: tuple[float, float],
    warp: WarpField,
    pixel_size_um: float,
    cfg: DetectionConfig,
    tile_index: int = -1,
    classifier: Classifier | None = None,
    id_start: int = 0,
    threshold: float = 0.5,
) -> list[Defect]:
    """Compare one tile. `prob`/`design` are the tile *core* arrays; `core_origin_px` is
    the level-0 px position of their (0,0)."""
    classifier = classifier or RuleClassifier(tolerance_px=cfg.tolerance_px)
    extra, missing = diff_masks(prob, design, threshold, cfg.tolerance_px)
    pred = prob > threshold
    dbool = design.astype(bool)
    min_px = max(cfg.min_area_px, int(round(cfg.min_area_um2 / pixel_size_um**2)))
    ox, oy = core_origin_px
    out: list[Defect] = []
    nid = id_start
    for kind, mask in ((EXTRA, extra), (MISSING, missing)):
        for blob, bb_local, area in blobs_from_mask(mask, min_px):
            cls, meta = ("unclassified", {})
            if cfg.classify:
                # classify in a local window for speed
                x0, y0, x1, y1 = bb_local.pad(8 + 2 * cfg.tolerance_px).to_int()
                x0, y0 = max(x0, 0), max(y0, 0)
                win = (slice(y0, y1), slice(x0, x1))
                ctx = BlobContext(blob[win], dbool[win], pred[win], pixel_size_um)
                cls, meta = classifier.classify(kind, ctx)
            score = float(np.abs(prob[blob] - design[blob].astype(np.float32)).mean())
            polys_px = [c + np.array([ox, oy]) for c in contours_px(blob)]
            polys_um = [warp.px_to_world(p) for p in polys_px]
            bb_px = BBox(bb_local.x0 + ox, bb_local.y0 + oy, bb_local.x1 + ox, bb_local.y1 + oy)
            allpts = np.vstack(polys_um) if polys_um else warp.px_to_world(bb_px.corners())
            bb_um = BBox.of_points(allpts)
            cy, cx = ndi.center_of_mass(blob)
            c_um = warp.px_to_world(np.array([[cx + ox + 0.5, cy + oy + 0.5]]))[0]
            out.append(
                Defect(
                    id=nid,
                    kind=kind,
                    cls=cls,
                    bbox_um=bb_um,
                    centroid_um=(float(c_um[0]), float(c_um[1])),
                    area_um2=float(area) * pixel_size_um**2,
                    area_px=area,
                    score=score,
                    polygons_um=polys_um,
                    bbox_px=bb_px,
                    tile_index=tile_index,
                    meta=meta,
                )
            )
            nid += 1
    return out


def merge_defects(defects: list[Defect], margin_um: float) -> list[Defect]:
    """Union defects of the same kind whose boxes touch within `margin_um` (tile-border splits)."""
    if not defects:
        return []
    n = len(defects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # sweep on x to avoid O(n^2) on big runs
    order = sorted(range(n), key=lambda i: defects[i].bbox_um.x0)
    active: list[int] = []
    for i in order:
        bi = defects[i].bbox_um
        active = [j for j in active if defects[j].bbox_um.x1 + margin_um >= bi.x0]
        for j in active:
            if defects[j].kind == defects[i].kind and defects[j].bbox_um.overlaps(bi, margin_um):
                parent[find(i)] = find(j)
        active.append(i)

    groups: dict[int, list[Defect]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(defects[i])
    merged: list[Defect] = []
    for k, (_, grp) in enumerate(sorted(groups.items())):
        if len(grp) == 1:
            d = grp[0]
            d.id = k
            merged.append(d)
            continue
        big = max(grp, key=lambda d: d.area_px)
        area_px = sum(d.area_px for d in grp)
        area_um2 = sum(d.area_um2 for d in grp)
        bb = grp[0].bbox_um
        bbpx = grp[0].bbox_px
        for d in grp[1:]:
            bb = bb.union(d.bbox_um)
            if bbpx is not None and d.bbox_px is not None:
                bbpx = bbpx.union(d.bbox_px)
        cx = sum(d.centroid_um[0] * d.area_px for d in grp) / area_px
        cy = sum(d.centroid_um[1] * d.area_px for d in grp) / area_px
        # a split blob that bridged two tiles may now deserve a different class; prefer the
        # most "serious" class seen among the parts
        cls = max((d.cls for d in grp), key=_severity)
        merged.append(
            Defect(
                id=k,
                kind=big.kind,
                cls=cls,
                bbox_um=bb,
                centroid_um=(cx, cy),
                area_um2=area_um2,
                area_px=area_px,
                score=float(np.mean([d.score for d in grp])),
                polygons_um=[p for d in grp for p in d.polygons_um],
                bbox_px=bbpx,
                tile_index=big.tile_index,
                meta={"merged_from": len(grp), **big.meta},
            )
        )
    return merged


_SEVERITY = {
    "short": 5,
    "open": 5,
    "missing_feature": 4,
    "extra_feature": 4,
    "particle": 3,
    "protrusion": 2,
    "notch": 2,
    "void": 1,
    "unclassified": 0,
}


def _severity(cls: str) -> int:
    return _SEVERITY.get(cls, 0)
