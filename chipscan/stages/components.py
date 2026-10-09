"""Component errors without ground truth: compare each component with its look-alikes in a line.

Josephson junctions, SQUIDs and test devices are placed many times, usually along a line
(a JJ chain, a TWPA unit cell row, a test array).  Nothing tells us what a perfect junction
looks like, but its neighbours do: within a line of the same device, the per-pixel median is
the reference and the per-pixel spread says how much variation is normal.  A member that
departs from both is an error; a regular line with a hole in it is a missing component.

Steps
  1. instances  - connected regions of the component mask (layout net or GDS), merged when
                  closer than a few px (a junction is two pads plus a link);
  2. patches    - one fixed-size crop per instance, size chosen per device type;
  3. lines      - instances of the same type sharing a row (or column) within a couple of px
                  after straightening, sorted by position; short lines fall back to the
                  most similar members of the same type anywhere on the scan;
  4. compare    - sub-pixel align to the line median, z = |diff| / (robust spread + noise),
                  keep connected areas above threshold -> outlined error;
  5. missing    - interior gaps of ~k x pitch in a regular line, confirmed by the image.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource


# ----------------------------------------------------------------------------- 1. instances
def find_instances(comp_map, threshold: int = 128, merge_px: int = 4, min_area_px: int = 20, chunk: int = 4096, is_class_map: bool = True) -> list[dict]:
    """Component instances from a uint8 map (class map == 2, or probability >= threshold)."""
    H, W = comp_map.shape
    out: list[dict] = []
    pad = 64
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            blk = np.asarray(comp_map[ya : min(H, y0 + chunk + pad), xa : min(W, x0 + chunk + pad)])
            m = (blk == 2) if is_class_map else (blk >= threshold)
            if not m.any():
                continue
            m = ndi.binary_opening(m, iterations=1)
            grown = ndi.binary_dilation(m, iterations=merge_px) if merge_px else m
            lab, _ = ndi.label(grown)
            lab = lab * m
            for i, sl in enumerate(ndi.find_objects(lab), start=1):
                if sl is None:
                    continue
                c = lab[sl] == i
                area = int(c.sum())
                if area < min_area_px:
                    continue
                cy, cx = ndi.center_of_mass(c)
                gx, gy = cx + sl[1].start + xa, cy + sl[0].start + ya
                if not (y0 <= gy < y0 + chunk and x0 <= gx < x0 + chunk):
                    continue
                out.append(
                    {
                        "cx": float(gx),
                        "cy": float(gy),
                        "w": int(sl[1].stop - sl[1].start),
                        "h": int(sl[0].stop - sl[0].start),
                        "area": area,
                        "bbox": [int(sl[1].start + xa), int(sl[0].start + ya), int(sl[1].stop + xa), int(sl[0].stop + ya)],
                    }
                )
    return out


# ----------------------------------------------------------------------------- 2. patches
def _type_key(inst: dict) -> tuple[int, int]:
    """Device type by footprint, quantised so the same device lands in the same bin."""
    q = lambda v: int(4 * round(v / 4.0))  # noqa: E731
    return q(inst["w"]), q(inst["h"])


def _patch_size(key: tuple[int, int]) -> int:
    return int(min(128, max(24, 8 * int(np.ceil((max(key) + 12) / 8.0)))))


def read_patches(source: ImageSource, centers: np.ndarray, size: int, block: int = 2048) -> np.ndarray:
    """(n, size, size) float32 crops, reading each block of the image once."""
    n = len(centers)
    out = np.zeros((n, size, size), np.float32)
    if n == 0:
        return out
    h = size // 2
    bx, by = (centers[:, 0] // block).astype(int), (centers[:, 1] // block).astype(int)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, k in enumerate(zip(bx, by, strict=True)):
        groups[k].append(i)
    for (gx, gy), idx in groups.items():
        x0, y0 = gx * block - h - 2, gy * block - h - 2
        win = source.read_region(x0, y0, x0 + block + size + 4, y0 + block + size + 4).astype(np.float32) / 255.0
        for i in idx:
            x, y = int(round(centers[i, 0])) - h - x0, int(round(centers[i, 1])) - h - y0
            out[i] = win[y : y + size, x : x + size]
    return out


# ----------------------------------------------------------------------------- 3. lines
def group_lines(insts: list[dict], idx: list[int], tol_px: float = 2.5, min_members: int = 5) -> list[list[int]]:
    """Rows and columns of same-type instances. Each instance joins its longer line."""
    pts = np.array([[insts[i]["cx"], insts[i]["cy"]] for i in idx])
    best: dict[int, list[int]] = {}
    for axis in (1, 0):  # rows share y, columns share x
        order = np.argsort(pts[:, axis])
        cur: list[int] = []
        last = None
        lines = []
        for o in order:
            v = pts[o, axis]
            if last is not None and v - last > tol_px:
                lines.append(cur)
                cur = []
            cur.append(o)
            last = v
        if cur:
            lines.append(cur)
        for ln in lines:
            if len(ln) < 2:
                continue
            members = [idx[o] for o in ln]
            for m in members:
                if m not in best or len(best[m]) < len(members):
                    best[m] = members
    seen, out = set(), []
    for ln in best.values():
        key = tuple(sorted(ln))
        if key in seen or len(ln) < min_members:
            continue
        seen.add(key)
        # sort along the line's long axis
        xs = [insts[i]["cx"] for i in ln]
        ys = [insts[i]["cy"] for i in ln]
        along = xs if np.ptp(xs) >= np.ptp(ys) else ys
        out.append([i for _, i in sorted(zip(along, ln, strict=True))])
    return out


def _split_by_look(ln: list[int], flat: np.ndarray, pos: dict[int, int], sim: float = 0.7, min_members: int = 5) -> list[list[int]]:
    """A row can hold several devices with the same footprint: split it into look-alike runs."""
    v = flat[[pos[i] for i in ln]]
    left = list(range(len(ln)))
    out = []
    while len(left) >= min_members:
        # seed with the member most similar to the others
        S = v[left] @ v[left].T
        seed = left[int(np.argmax(np.median(S, axis=1)))]
        grp = [k for k in left if float(v[k] @ v[seed]) > sim]
        if len(grp) < min_members:
            break
        out.append(grp)
        left = [k for k in left if k not in grp]
    # leftovers sit in the row but look different: a damaged member must still be judged
    # against its row, so attach it to the run it resembles most (if it resembles any at all)
    if out:
        means = [v[g].mean(0) for g in out]
        means = [m / (np.linalg.norm(m) + 1e-9) for m in means]
        for k in left:
            sims = [float(v[k] @ m) for m in means]
            j = int(np.argmax(sims))
            if sims[j] > 0.25:
                out[j].append(k)
    return [sorted(ln[k] for k in g) for g in out]


# ----------------------------------------------------------------------------- 4. compare
def _align_to(ref: np.ndarray, p: np.ndarray, max_shift: float = 3.0) -> np.ndarray:
    from skimage.registration import phase_cross_correlation

    if p.std() < 1e-3 or ref.std() < 1e-3:
        return p
    try:
        s, _, _ = phase_cross_correlation(ref, p, upsample_factor=8, normalization=None)
    except Exception:
        return p
    if np.hypot(*s) > max_shift:
        return p
    return ndi.shift(p, s, order=1, mode="nearest")


def compare_group(patches: np.ndarray, z_min: float = 6.0, min_diff: float = 0.06, min_area: int = 4) -> list[dict]:
    """Per member: outlier mask + scores against the group's median and spread."""
    n = len(patches)
    if n < 3:
        return [{"z": 0.0, "ncc": 1.0, "mask": None} for _ in range(n)]
    # normalise brightness (exposure varies along a scan), align to a first median, recompute
    norm = patches - np.median(patches, axis=(1, 2), keepdims=True)
    med = np.median(norm, axis=0)
    al = np.stack([_align_to(med, p) for p in norm])
    med = np.median(al, axis=0)
    spread = 1.4826 * np.median(np.abs(al - med), axis=0)
    noise = max(1.4826 * float(np.median(np.abs(al - med))), 0.006)
    out = []
    mc = med - med.mean()
    # the median/MAD of a small group is itself noisy: demand more evidence there
    z_eff = z_min * (1.0 + 8.0 / n)
    for p in al:
        d = ndi.gaussian_filter(p - med, 0.8)
        z = np.abs(d) / (np.maximum(spread, noise) + noise)
        m = (z > z_eff) & (np.abs(d) > min_diff)
        lab, k = ndi.label(m)
        if k:
            sizes = ndi.sum(m, lab, range(1, k + 1))
            m = np.isin(lab, 1 + np.nonzero(sizes >= min_area)[0])
        pc = p - p.mean()
        ncc = float((pc * mc).sum() / max(np.sqrt((pc * pc).sum() * (mc * mc).sum()), 1e-9))
        out.append({"z": float(z[m].max()) if m.any() else float(np.percentile(z, 99.5)), "ncc": ncc, "mask": m if m.any() else None})
    return out


# ----------------------------------------------------------------------------- 5. pipeline
def component_errors(
    source: ImageSource,
    insts: list[dict],
    pixel_size_um: float | None = None,
    z_min: float = 6.0,
    min_ncc: float = 0.55,
    min_type_members: int = 6,
    progress=None,
    id_start: int = 0,
) -> tuple[list[dict], dict]:
    """Outlined component errors and a summary (types, lines, members)."""
    px = pixel_size_um or 1.0
    by_type: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, d in enumerate(insts):
        by_type[_type_key(d)].append(i)
    recs: list[dict] = []
    nid = id_start
    summary = {"n_components": len(insts), "n_types": 0, "n_lines": 0, "n_compared": 0}
    H, W = source.shape

    def inside(d: dict, size: int) -> bool:
        h = size // 2 + 2
        return h <= d["cx"] < W - h and h <= d["cy"] < H - h

    by_type = {k: [i for i in v if inside(insts[i], _patch_size(k))] for k, v in by_type.items()}
    types = [(k, v) for k, v in by_type.items() if len(v) >= min_type_members]
    summary["n_types"] = len(types)
    for ti, (key, idx) in enumerate(types):
        size = _patch_size(key)
        centers = np.array([[insts[i]["cx"], insts[i]["cy"]] for i in idx])
        P = read_patches(source, centers, size)
        pos = {g: k for k, g in enumerate(idx)}
        flat = P.reshape(len(P), -1)
        flat = flat - flat.mean(1, keepdims=True)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
        lines = []
        for ln in group_lines(insts, idx):
            lines += _split_by_look(ln, flat, pos)
        in_line = {i for ln in lines for i in ln}
        summary["n_lines"] += len(lines)
        groups = [(ln, "line") for ln in lines]
        # members not on a usable line: compare with the most similar members of their type
        rest = [i for i in idx if i not in in_line]
        for i in rest:
            sims = flat @ flat[pos[i]]
            sims[pos[i]] = -1
            nn = np.argsort(sims)[::-1][:24]
            # a one-of-a-kind device has no reference: it cannot be judged, so it is not flagged
            if (sims[nn] > 0.75).sum() < 5:
                continue
            nn = [j for j in nn if sims[j] > 0.75]
            groups.append(([idx[j] for j in nn] + [i], "type"))
        for members, how in groups:
            res = compare_group(P[[pos[m] for m in members]], z_min=z_min)
            targets = members if how == "line" else members[-1:]
            res_t = res if how == "line" else res[-1:]
            summary["n_compared"] += len(targets)
            for m, r in zip(targets, res_t, strict=True):
                bad_local = r["mask"] is not None
                bad_global = r["ncc"] < min_ncc
                if not (bad_local or bad_global):
                    continue
                d = insts[m]
                h = size // 2
                ox, oy = int(round(d["cx"])) - h, int(round(d["cy"])) - h
                mask = r["mask"] if bad_local else np.zeros((size, size), bool)
                if bad_global and not bad_local:  # whole device differs: outline its footprint
                    x0, y0, x1, y1 = d["bbox"]
                    mask[max(0, y0 - oy) : y1 - oy, max(0, x0 - ox) : x1 - ox] = True
                cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [(c.reshape(-1, 2).astype(float) + [ox + 0.5, oy + 0.5]).round(2).tolist() for c in cs if len(c) >= 3]
                ys, xs = np.nonzero(mask)
                area = int(mask.sum())
                bb = {"x0": float(xs.min() + ox), "y0": float(ys.min() + oy), "x1": float(xs.max() + 1 + ox), "y1": float(ys.max() + 1 + oy)}
                cxx, cyy = float(xs.mean() + ox), float(ys.mean() + oy)
                recs.append(
                    {
                        "id": nid,
                        "group": "component",
                        "kind": "component",
                        "class": "component_deformed" if bad_global and not bad_local else "component_defect",
                        "area_px": area,
                        "score": float(min(1.0, max(r["z"] / 20.0, 1.0 - r["ncc"]))),
                        "z": r["z"],
                        "ncc": r["ncc"],
                        "n_ref": len(members),
                        "reference": how,
                        "component_bbox_px": {"x0": d["bbox"][0], "y0": d["bbox"][1], "x1": d["bbox"][2], "y1": d["bbox"][3]},
                        "bbox_px": bb,
                        "centroid_px": [cxx, cyy],
                        "polygons_px": polys,
                        "area_um2": area * px * px,
                        "centroid_um": [cxx * px, cyy * px],
                        "bbox_um": {k: v * px for k, v in bb.items()},
                    }
                )
                nid += 1
        # missing members in regular lines
        for ln in lines:
            recs_m = _missing_in_line(source, insts, ln, P, pos, size, px, nid)
            nid += len(recs_m)
            recs.extend(recs_m)
        if progress:
            progress(ti + 1, len(types))
    return recs, summary


def _missing_in_line(source, insts, ln, P, pos, size, px, nid) -> list[dict]:
    if len(ln) < 8:
        return []
    xy = np.array([[insts[i]["cx"], insts[i]["cy"]] for i in ln])
    ax = 0 if np.ptp(xy[:, 0]) >= np.ptp(xy[:, 1]) else 1
    gaps = np.diff(xy[:, ax])
    pitch = float(np.median(gaps))
    if pitch < 4 or np.mean(np.abs(gaps - pitch) < 0.08 * pitch) < 0.7:  # not a regular line
        return []
    med = np.median(P[[pos[i] for i in ln]], axis=0)
    mc = med - med.mean()
    out = []
    for k, g in enumerate(gaps):
        m = g / pitch
        if m < 1.6 or abs(m - round(m)) > 0.15:
            continue
        for j in range(1, int(round(m))):
            e = xy[k] + (xy[k + 1] - xy[k]) * j / round(m)
            patch = read_patches(source, e[None], size)[0]
            pc = patch - patch.mean()
            ncc = float((pc * mc).sum() / max(np.sqrt((pc * pc).sum() * (mc * mc).sum()), 1e-9))
            if ncc > 0.5:  # it is there after all (the mask just missed it)
                continue
            ref = insts[ln[k]]
            w, h = ref["w"], ref["h"]
            bb = {"x0": float(e[0] - w / 2), "y0": float(e[1] - h / 2), "x1": float(e[0] + w / 2), "y1": float(e[1] + h / 2)}
            poly = [[bb["x0"], bb["y0"]], [bb["x1"], bb["y0"]], [bb["x1"], bb["y1"]], [bb["x0"], bb["y1"]]]
            out.append(
                {
                    "id": nid + len(out),
                    "group": "component",
                    "kind": "component",
                    "class": "component_missing",
                    "area_px": int(w * h),
                    "score": float(min(1.0, 1.0 - ncc)),
                    "ncc": ncc,
                    "n_ref": len(ln),
                    "reference": "line",
                    "bbox_px": bb,
                    "centroid_px": [float(e[0]), float(e[1])],
                    "polygons_px": [poly],
                    "area_um2": w * h * px * px,
                    "centroid_um": [float(e[0]) * px, float(e[1]) * px],
                    "bbox_um": {k2: v * px for k2, v in bb.items()},
                }
            )
    return out
