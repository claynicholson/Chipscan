"""Classical faint-speck finder (design-free).

Dust and haze on a wafer are small, soft blobs of a few grey levels on an otherwise
flat background: too faint for a learned detector trained on visible defects, but
easy to find statistically.  For each tile we estimate the local background and
noise, apply difference-of-Gaussian blob filters at several scales, and keep blobs
whose *area-integrated* signal exceeds `z_min` sigma.  Pixels near strong designed
edges are excluded (edges ring in a DoG).  Output records have the same shape as
the learned anomalies so they merge into the same list, with kind "speck".
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource


def find_specks_tile(
    img: np.ndarray,
    sigmas: tuple[float, ...] = (2.5, 4.0, 7.0),
    z_min: float = 7.0,
    bg_sigma: float = 25.0,
    edge_exclude_px: int | None = None,
    min_area_px: int = 10,
    max_area_px: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask, zmap) for one float32 tile in [0,1]."""
    import cv2

    f = img.astype(np.float32)
    # median background: does not bleed designed structure into its surroundings the way a Gaussian does
    k = int(2 * bg_sigma) | 1
    bg = cv2.medianBlur(np.clip(f * 255, 0, 255).astype(np.uint8), k).astype(np.float32) / 255.0
    resid = f - bg
    # robust noise estimate of the residual
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
    # designed edges: strong gradients in a lightly smoothed image
    sm = ndi.gaussian_filter(f, 1.5)
    gmag = np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 1))
    edge = (gmag > max(8.0 * noise, 0.03)) | (np.abs(resid) > max(12.0 * noise, 0.06))  # strong edges and gross structure
    # the DoG of the largest scale rings ~3 sigma beyond a strong edge: exclude that far
    excl = edge_exclude_px if edge_exclude_px is not None else int(np.ceil(3.0 * max(sigmas)))
    edge = ndi.binary_dilation(edge, iterations=excl)
    zbest = np.zeros_like(f)
    for s in sigmas:
        # DoG blob filter; its noise std is noise * ||kernel||_2 which we estimate numerically
        dog = ndi.gaussian_filter(resid, s) - ndi.gaussian_filter(resid, s * 1.6)
        k = np.zeros((int(8 * s * 1.6) | 1,) * 2, np.float32)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
        kd = ndi.gaussian_filter(k, s) - ndi.gaussian_filter(k, s * 1.6)
        dog_noise = noise * float(np.sqrt((kd**2).sum()))
        z = np.abs(dog) / dog_noise
        # a DoG rings with the opposite sign around any blob or edge; keep only responses whose
        # sign agrees with the smoothed residual itself (a dark speck must look dark)
        agree = dog * ndi.gaussian_filter(resid, s) > 0
        z[~agree] = 0.0
        zbest = np.maximum(zbest, z)
    zbest[edge] = 0.0
    mask = zbest > z_min
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
        bad = np.isin(lab, np.nonzero((sizes < min_area_px) | (sizes > max_area_px))[0] + 1)
        mask &= ~bad
    return mask, zbest


def find_specks(source: ImageSource, tile_px: int = 1024, margin_px: int = 64, z_min: float = 7.0, pixel_size_um: float | None = None, id_start: int = 0, progress: bool = True) -> list[dict]:
    """Run the speck finder over a whole image; returns anomaly-style records (kind 'speck')."""
    import cv2
    from rich.progress import Progress

    from chipscan.tiling import make_tiles

    tiles = make_tiles(source.shape, tile_px, margin_px)
    recs: list[dict] = []
    nid = id_start
    prog = Progress() if progress else None
    if prog:
        prog.start()
        task = prog.add_task("faint specks", total=len(tiles))
    H, W = source.shape
    for t in tiles:
        x0, y0, x1, y1 = t.read.to_int()
        # clamp to the image: zero padding would fake a huge edge and inflate the noise estimate
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        img = source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0
        if img.std() < 1e-3:
            if prog:
                prog.advance(task)
            continue
        mask, z = find_specks_tile(img, z_min=z_min)
        # the image border itself rings: ignore a band along the global edge
        b = 24
        if x0 == 0:
            mask[:, :b] = False
        if y0 == 0:
            mask[:b, :] = False
        if x1 == W:
            mask[:, -b:] = False
        if y1 == H:
            mask[-b:, :] = False
        cy0, cx0_ = int(t.core.y0) - y0, int(t.core.x0) - x0
        sl = (slice(cy0, cy0 + int(t.core.height)), slice(cx0_, cx0_ + int(t.core.width)))
        core, zc = mask[sl], z[sl]
        if core.any():
            lab, n = ndi.label(core)
            cx0, cy0 = int(t.core.x0), int(t.core.y0)
            for i, s in enumerate(ndi.find_objects(lab), start=1):
                if s is None:
                    continue
                comp = lab[s] == i
                area = int(comp.sum())
                cy, cx = ndi.center_of_mass(comp)
                gx, gy = cx + s[1].start + cx0, cy + s[0].start + cy0
                cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [(c.reshape(-1, 2).astype(float) + [s[1].start + cx0 + 0.5, s[0].start + cy0 + 0.5]).round(2).tolist() for c in cs if len(c) >= 3]
                bb = {"x0": float(s[1].start + cx0), "y0": float(s[0].start + cy0), "x1": float(s[1].stop + cx0), "y1": float(s[0].stop + cy0)}
                px = pixel_size_um or 1.0
                recs.append(
                    {
                        "id": nid,
                        "kind": "speck",
                        "class": "speck",
                        "area_px": area,
                        "score": float(min(1.0, zc[s][comp].max() / 20.0)),
                        "z": float(zc[s][comp].max()),
                        "bbox_px": bb,
                        "centroid_px": [float(gx), float(gy)],
                        "polygons_px": polys,
                        "area_um2": area * px**2,
                        "centroid_um": [float(gx) * px, float(gy) * px],
                        "bbox_um": {k: v * px for k, v in bb.items()},
                    }
                )
                nid += 1
        if prog:
            prog.advance(task)
    if prog:
        prog.stop()
    return recs
