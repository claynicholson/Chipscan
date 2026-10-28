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


def local_feature(source, d: dict, px: float, reach_px: int = 60) -> dict | None:
    """Width and kind of the gap or line a finding sits in, from profiles that pass *beside* it.

    The dominant edge direction around the finding decides the profile direction (across the
    feature); profiles are taken just before and after the finding along the feature, so the
    finding itself does not disturb them.  Edges are the dark outline minima; the feature is the
    stretch between the nearest edge on each side of the finding's centre.  Darker than its
    surroundings = gap (substrate), brighter = line (metal)."""
    from chipscan.stages.cpw import edge_dips

    b = d["bbox_px"]
    cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
    hw, hh = (b["x1"] - b["x0"]) / 2 + 3, (b["y1"] - b["y0"]) / 2 + 3
    R = int(reach_px + max(hw, hh))
    x0, y0 = int(cx - R), int(cy - R)
    win = source.read_region(x0, y0, x0 + 2 * R, y0 + 2 * R).astype(np.float64)
    if win.ndim == 3:
        win = win.mean(-1)
    gx, gy = ndi.sobel(ndi.gaussian_filter(win, 1.0), 1), ndi.sobel(ndi.gaussian_filter(win, 1.0), 0)
    m = np.ones_like(win, bool)
    m[int(R - hh) : int(R + hh), int(R - hw) : int(R + hw)] = False  # not the finding itself
    jxx, jyy = float((gx * gx)[m].sum()), float((gy * gy)[m].sum())
    across_x = jxx >= jyy  # strong x-gradients: vertical feature, profile runs along x
    widths = []
    for off in (-(hh if across_x else hw) - 4, (hh if across_x else hw) + 4):
        if across_x:
            r = int(round(R + off))
            if not (2 <= r < 2 * R - 2):
                continue
            prof = win[r - 2 : r + 3].mean(0)
            c = R
        else:
            cc = int(round(R + off))
            if not (2 <= cc < 2 * R - 2):
                continue
            prof = win[:, cc - 2 : cc + 3].mean(1)
            c = R
        e = edge_dips(prof, min_depth=5.0)
        left, right = e[e < c], e[e > c]
        if not len(left) or not len(right):
            continue
        a_, b_ = left.max(), right.min()
        inner = prof[int(np.ceil(a_)) + 1 : int(b_)]
        outer = np.r_[prof[max(0, int(a_) - 6) : max(0, int(a_) - 1)], prof[int(b_) + 2 : int(b_) + 7]]
        if len(inner) < 1 or len(outer) < 2:
            continue
        widths.append((b_ - a_, float(inner.mean() - outer.mean())))
    if not widths:
        return None
    w = float(np.median([x[0] for x in widths])) * px
    contrast = float(np.median([x[1] for x in widths]))
    if w > 200:
        return None
    return {"kind": "gap" if contrast < 0 else "line", "width_um": round(w, 2), "orientation": "vertical" if across_x else "horizontal"}


def assess(findings: list[dict], class_map, pixel_size_um: float | None, window_um: float = 80.0, source=None) -> list[dict]:
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
        # the feature (gap or line) the finding sits in, measured from the image beside it
        gap_w, bridges, feat = None, False, None
        if source is not None and d.get("group") == "speck":
            feat = local_feature(source, d, px)
            if feat is not None:
                ys_, xs_ = np.nonzero(own)
                extent = float(max(np.ptp(xs_) + 1, np.ptp(ys_) + 1) * px) if len(xs_) else size
                if feat["kind"] == "gap":
                    gap_w = feat["width_um"]
                    # it can short the gap when it reaches across it
                    bridges = extent >= 0.8 * gap_w
        d["feature"] = feat
        d["gap_um"] = round(gap_w, 2) if gap_w else None
        d["bridges_gap"] = bool(bridges)
        jj = float(np.exp(-d_jj / L_JJ_UM))
        tr = float(np.exp(-d_tr / L_TRACE_UM))
        sz = float(1.0 - np.exp(-size / SIZE_UM))
        g = d.get("group", "speck")
        conf = float(np.clip(d.get("score", 0.5), 0, 1))
        if g == "component":
            # it is on a junction by construction; how bad depends on how different it is
            cls_name = d.get("class", "")
            if cls_name == "component_missing":
                r = 0.85
            else:
                diff = max(min(1.0, (d.get("z") or 0) / 40.0), min(1.0, max(0.0, 1.0 - (d.get("ncc") or 1.0))))
                r = 0.3 + 0.6 * diff + 0.1 * sz
        elif g == "trace":
            # a shape error is on a trace by construction; near a junction it matters most
            # x0.7: the trace detector is the least reliable of the three (validated precision ~0.6),
            # so an unconfirmed trace finding should not outrank dirt or a component error on a JJ
            r = 0.7 * (0.45 + 0.35 * jj + 0.2 * sz) * (0.6 + 0.4 * conf)
        else:  # speck
            # physical failure modes, worst one wins:
            #   short    - reaches across a gap between two conductors (any size that spans it)
            #   junction - near a JJ: shifts its critical current / adds loss at the junction
            #   loss     - in or at a gap: lossy dielectric where the field is
            #   cosmetic - anywhere else, by size
            p_short = 0.95 if bridges else 0.0
            p_jj = jj * (0.35 + 0.65 * sz)
            in_gap = bool(feat and feat["kind"] == "gap")
            p_loss = (0.55 if in_gap else 0.25) * tr * (0.3 + 0.7 * sz)  # the field lives in the gaps
            p_floor = 0.12 * sz
            r = max(p_short, p_jj, p_loss, p_floor) * (0.7 + 0.3 * conf)
            d["failure_mode"] = ["short", "junction", "loss", "cosmetic"][int(np.argmax([p_short, p_jj, p_loss, p_floor]))]
            if d.get("class") == "probe_mark":
                r *= 0.6  # pads are made to be probed: a scrub there matters only if it is large
        if d.get("verified") is False:
            r *= 0.85  # no twin to compare with (unique structure): less certain it is a defect
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


def suppress_component_shapes(findings: list[dict], class_map, min_iou: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Drop speck candidates whose outline is a component itself (a junction the detector mistook
    for dirt). Dirt *on* a junction overlaps only part of it and survives."""
    kept, dropped = [], []
    H, W = class_map.shape
    for d in findings:
        b = d["bbox_px"]
        x0, y0 = int(max(0, b["x0"] - 6)), int(max(0, b["y0"] - 6))
        x1, y1 = int(min(W, b["x1"] + 6)), int(min(H, b["y1"] + 6))
        cls = np.asarray(class_map[y0:y1, x0:x1])
        own = _finding_mask(d, x0, y0, cls.shape)
        comp = ndi.binary_dilation(cls == 2, iterations=2)
        lab, _ = ndi.label(comp)
        ids = np.unique(lab[own & (lab > 0)])
        if len(ids):
            c = np.isin(lab, ids)
            iou = float((c & own).sum()) / max(1, float((c | own).sum()))
            if iou >= min_iou:
                dropped.append(d)
                continue
        kept.append(d)
    return kept, dropped


def suppress_text_lines(findings: list[dict], source, window: int = 160) -> tuple[list[dict], list[dict]]:
    """Drop detections that are one glyph in a row of glyphs: chip labels in any font size.

    Around each detection, find the blobs that differ from the background; if the blob under the
    detection has at least three neighbours of similar height, aligned on one baseline and spaced
    like characters, it is lettering, whatever the layout net called it."""
    import cv2  # noqa: F401

    from chipscan.stages.inject import Tones

    kept, dropped = [], []
    for d in findings:
        cx, cy = (int(round(v)) for v in d["centroid_px"])
        x0, y0 = cx - window, cy - window
        win8 = np.ascontiguousarray(source.read_region(x0, y0, cx + window, cy + window), dtype=np.uint8)
        win = win8.astype(np.float32) / 255.0
        t = Tones(win)
        # contrast against the *local* background: lettering often sits on its own darker panel
        # (OpenCV's histogram median: ~1000x faster than scipy's at this size)
        local = cv2.medianBlur(win8, 41).astype(np.float32) / 255.0
        ink = np.abs(win - local) > max(0.05, 4 * t.noise)
        blobs = ndi.binary_closing(ink, iterations=1)
        lab, n = ndi.label(blobs)
        if n < 4:
            kept.append(d)
            continue
        objs = ndi.find_objects(lab)
        # the glyph under the detection: the smallest character-sized blob near its centre
        near = lab[window - 6 : window + 7, window - 6 : window + 7]
        cands = [int(v) for v in np.unique(near[near > 0])]
        cands = [c for c in cands if objs[c - 1] is not None and 6 <= objs[c - 1][0].stop - objs[c - 1][0].start <= 120]
        if not cands:
            kept.append(d)
            continue
        me = min(cands, key=lambda c: (objs[c - 1][0].stop - objs[c - 1][0].start) * (objs[c - 1][1].stop - objs[c - 1][1].start))
        sl = objs[me - 1]
        h0 = sl[0].stop - sl[0].start
        w0 = sl[1].stop - sl[1].start
        if not (6 <= h0 <= 120):
            kept.append(d)
            continue
        # touching characters merge into one word-shaped blob: wide, character height, full of
        # enclosed holes (outlined and closed letters); dirt has none of that structure
        mine = lab[sl] == me
        holes = ndi.label(ndi.binary_fill_holes(mine) & ~mine)[1]
        # a line of characters: crossing the word's middle rows hits ink many times (strokes)
        mid = (ink[sl] & mine)[max(0, h0 // 2 - 2) : h0 // 2 + 3]
        crossings = int(np.median([np.count_nonzero(np.diff(r.astype(np.int8)) == 1) for r in mid])) if mid.size else 0
        # a particle lying on a trace merges with the line into one wide holey blob: a word never
        # has one row of unbroken ink spanning (nearly) its whole width
        run = 0
        for r_ in (ink[sl] & mine):
            if r_.any():
                e = np.flatnonzero(np.diff(np.r_[0, r_.astype(np.int8), 0]))
                run = max(run, int((e[1::2] - e[::2]).max()))
        on_line = run >= 0.7 * w0
        if w0 >= 1.8 * h0 and not on_line and (holes >= 3 or (12 <= h0 <= 80 and crossings >= 5)):
            dropped.append(d)
            continue
        if w0 > 3 * h0:
            kept.append(d)
            continue
        base0 = sl[0].stop
        # lettering is printed in one tone: fellow glyphs share polarity and contrast with this one
        # (a dark particle beside a row of bright junction pads is not a word)
        sig = win - local
        c0 = float(np.median(sig[sl][mine]))
        mates = 0
        for j, o in enumerate(objs, start=1):
            if o is None or o is sl:
                continue
            cj = float(np.median(sig[o][lab[o] == j]))
            if np.sign(cj) != np.sign(c0) or not (0.5 <= abs(cj) / max(abs(c0), 1e-6) <= 2.0):
                continue
            h = o[0].stop - o[0].start
            w = o[1].stop - o[1].start
            if abs(h - h0) <= 0.3 * h0 and abs(o[0].stop - base0) <= 0.25 * h0 and w <= 2.5 * h0:
                gap = min(abs(o[1].start - sl[1].stop), abs(sl[1].start - o[1].stop))
                if gap <= 4 * h0:
                    mates += 1
        (dropped if mates >= 3 else kept).append(d)
    return kept, dropped


def suppress_text(findings: list[dict], class_map, frac: float = 0.4, grow_px: int = 4, keep_compact: bool = False) -> tuple[list[dict], list[dict]]:
    """Drop findings that lie mostly on text (labels, test-structure annotations).

    With `keep_compact`, solid blobs survive even inside a "text" region: the layout net tends
    to call dark particles text, while what a detector picks up on real lettering is a thin
    sliver along a stroke."""
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
        if on >= frac and keep_compact and own.sum() >= 30 and ndi.binary_erosion(own, iterations=2).any():
            on = 0.0
        (dropped if on >= frac else kept).append(d)
    return kept, dropped
