"""Compare the scan with the design, tile by tile: extra metal (shorts) and missing metal (opens).

Each tile is first aligned to the rendered design locally (edge phase correlation, so residual
stitching offsets do not matter), then the image is split into its two process tones - metal
(bright) and exposed substrate - with a threshold between the tile's own two modes.  Pixels within
`tol_px` of a design edge are ignored (edge placement is metrology's job, not a defect).  What is
left and disagrees with the design, in blobs above `min_area_um2`, is reported:
  extra_metal   - metal where the design has a gap (bridges, shorts, residue of metal)
  missing_metal - substrate where the design has metal (opens, nicks, lifted metal)
Dark particles are neither tone and are left to the speck detector.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from chipscan.design.wafer import WaferDesign
from chipscan.design.wafer_register import WaferRegistration
from chipscan.io.readers import ImageSource
from chipscan.registration.coarse import edge_map


def compare_tile(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, c: int, r: int, n: int = 1024, tol_px: int = 3, min_area_um2: float = 40.0, max_shift: int = 40):
    img = src.read_region(c, r, c + n, r + n).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(-1)
    d = reg.render_window(wd, c, r, n, n).astype(np.float32)
    if d.std() < 0.05 or img.std() < 2:
        return [], None
    from skimage.registration import phase_cross_correlation

    sh, _, _ = phase_cross_correlation(edge_map(img / 255.0, 1.2), edge_map(d, 1.2), upsample_factor=4, normalization="phase")
    if abs(sh[0]) > max_shift or abs(sh[1]) > max_shift:
        sh = (0.0, 0.0)
    d = cv2.warpAffine(d, np.float32([[1, 0, sh[1]], [0, 1, sh[0]]]), (n, n), flags=cv2.INTER_NEAREST) > 0.5
    # the two process tones: compare each pixel with the design's own metal / gap tone in this tile
    sm = ndi.median_filter(img, size=3)
    m_t, g_t = np.median(sm[ndi.binary_erosion(d, iterations=4)]) if d.sum() > 500 else None, np.median(sm[ndi.binary_erosion(~d, iterations=4)]) if (~d).sum() > 500 else None
    if m_t is None or g_t is None or abs(m_t - g_t) < 8:
        return [], sh
    thr = (m_t + g_t) / 2
    metal = (sm > thr) if m_t > g_t else (sm < thr)
    # not a process tone (much darker than both: particles, edge outlines) -> not judged here
    lo = min(m_t, g_t) - 0.6 * abs(m_t - g_t)
    judged = sm > lo
    band = ndi.binary_dilation(d ^ ndi.binary_erosion(d), iterations=tol_px)
    px = reg.pixel_size_um
    out = []
    for kind, diff in (("extra_metal", metal & ~d), ("missing_metal", ~metal & d)):
        diff = ndi.binary_opening(diff & judged & ~band, iterations=1)
        lab, k = ndi.label(diff)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            blob = lab[sl] == i
            a = blob.sum() * px * px
            if a < min_area_um2:
                continue
            ys, xs = np.nonzero(blob)
            out.append({"kind": kind, "area_um2": round(float(a), 1), "centroid_px": [float(c + sl[1].start + xs.mean()), float(r + sl[0].start + ys.mean())],
                        "bbox_px": {"x0": float(c + sl[1].start), "y0": float(r + sl[0].start), "x1": float(c + sl[1].stop), "y1": float(r + sl[0].stop)}})
    return out, sh
