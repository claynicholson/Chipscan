"""Classical (no-training) segmentation baselines.

* ``otsu``         : global Otsu threshold per tile.
* ``adaptive``     : local mean threshold (handles illumination drift across a mosaic).
* ``design_prior`` : fit the two intensity classes *using the registered design as a
                     prior* (mean/std of pixels the design says are metal vs. substrate,
                     away from edges), then threshold at the Bayes boundary.  This is the
                     strongest classical baseline and is what the UNet must beat.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


class ClassicalSegmenter:
    uses_design_prior = True

    def __init__(
        self,
        method: str = "design_prior",
        metal_bright: bool = True,
        blur_sigma: float = 1.0,
        adaptive_block_px: int = 101,
        morph_px: int = 0,
        edge_exclusion_px: int = 3,
    ):
        self.name = f"classical:{method}"
        self.method = method
        self.metal_bright = metal_bright
        self.blur_sigma = blur_sigma
        self.adaptive_block_px = adaptive_block_px | 1
        self.morph_px = morph_px
        self.edge_exclusion_px = edge_exclusion_px
        self.global_stats: tuple[tuple[float, float], tuple[float, float]] | None = None
        """((metal mean, std), (substrate mean, std)) over the whole image; fallback for tiles
        that contain only one class, where a per-tile threshold is ill-posed."""

    def fit_global(self, source, design, warp, region, n_tiles: int = 24, tile_px: int = 512, seed: int = 0) -> None:
        """Estimate image-wide class statistics from tiles sampled over `region` (level-0 px)."""
        import numpy as _np

        from chipscan.geometry import BBox as _BBox

        rng = _np.random.default_rng(seed)
        ms, ss = [], []
        for _ in range(n_tiles * 3):
            if len(ms) >= n_tiles:
                break
            x = int(rng.uniform(region.x0, max(region.x1 - tile_px, region.x0)))
            y = int(rng.uniform(region.y0, max(region.y1 - tile_px, region.y0)))
            bb = _BBox(x, y, x + tile_px, y + tile_px)
            img = source.read_region(x, y, x + tile_px, y + tile_px, level=0).astype(_np.float32) / 255.0
            prior = design.render(warp.local_affine(bb), bb).astype(bool)
            t = self._prep(img)
            if not self.metal_bright:
                t = 1.0 - t
            edge = prior ^ ndi.binary_erosion(prior, iterations=1)
            far = ~ndi.binary_dilation(edge, iterations=self.edge_exclusion_px)
            m, sub = t[prior & far], t[(~prior) & far]
            if m.size > 200 and sub.size > 200:
                ms.append((float(_np.median(m)), float(m.std())))
                ss.append((float(_np.median(sub)), float(sub.std())))
        if ms:
            self.global_stats = (
                (float(_np.median([a for a, _ in ms])), float(_np.median([b for _, b in ms]))),
                (float(_np.median([a for a, _ in ss])), float(_np.median([b for _, b in ss]))),
            )

    def _prep(self, t: np.ndarray) -> np.ndarray:
        if self.blur_sigma > 0:
            t = cv2.GaussianBlur(t, (0, 0), self.blur_sigma)
        return t

    def _one(self, tile: np.ndarray, prior: np.ndarray | None) -> np.ndarray:
        t = self._prep(tile.astype(np.float32))
        if not self.metal_bright:
            t = 1.0 - t
        if self.method == "otsu":
            if self.global_stats is not None and (prior is None or prior.min() == prior.max()):
                (gm, gsm), (gs, gss) = self.global_stats
                prob = _soft(t, 0.5 * (gm + gs), max(0.5 * (gsm + gss), 1e-3))
            else:
                thr = threshold_otsu(t) if t.std() > 1e-4 else 0.5
                prob = _soft(t, thr, max(t.std(), 1e-3) * 0.5)
        elif self.method == "adaptive":
            local = cv2.blur(t, (self.adaptive_block_px, self.adaptive_block_px))
            prob = _soft(t - local, 0.0, max(t.std(), 1e-3) * 0.5)
        elif self.method == "design_prior":
            if prior is None or prior.min() == prior.max():
                # single-class tile (bare substrate, or solid ground plane): a per-tile threshold is
                # ill-posed, so use the image-wide class statistics when we have them
                if self.global_stats is not None:
                    (gm, gsm), (gs, gss) = self.global_stats
                    prob = _soft(t, 0.5 * (gm + gs), max(0.5 * (gsm + gss), 1e-3))
                else:
                    thr = threshold_otsu(t) if t.std() > 1e-4 else 0.5
                    prob = _soft(t, thr, max(t.std(), 1e-3) * 0.5)
            else:
                p = prior.astype(bool)
                edge = p ^ ndi.binary_erosion(p, iterations=1)
                far = ~ndi.binary_dilation(edge, iterations=self.edge_exclusion_px)
                m, s = t[p & far], t[(~p) & far]
                if m.size < 10 or s.size < 10:
                    if self.global_stats is not None:
                        (gm, gsm), (gs, gss) = self.global_stats
                        prob = _soft(t, 0.5 * (gm + gs), max(0.5 * (gsm + gss), 1e-3))
                    else:
                        thr = threshold_otsu(t) if t.std() > 1e-4 else 0.5
                        prob = _soft(t, thr, max(t.std(), 1e-3) * 0.5)
                else:
                    mu1, mu0 = float(m.mean()), float(s.mean())
                    sd = max(0.5 * (float(m.std()) + float(s.std())), 1e-3)
                    if mu1 < mu0:  # polarity in config was wrong for this tile; trust the data
                        mu1, mu0 = mu0, mu1
                        t = 1.0 - t
                        mu1, mu0 = 1.0 - mu1, 1.0 - mu0
                    thr = 0.5 * (mu1 + mu0)
                    prob = _soft(t, thr, sd)
        else:
            raise ValueError(f"unknown classical method {self.method!r}")
        if self.morph_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_px * 2 + 1,) * 2)
            b = (prob > 0.5).astype(np.uint8)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
            prob = np.where(b > 0, np.maximum(prob, 0.75), np.minimum(prob, 0.25)).astype(np.float32)
        return prob.astype(np.float32)

    def predict(self, tiles: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
        out = np.empty(tiles.shape, dtype=np.float32)
        for i in range(tiles.shape[0]):
            out[i] = self._one(tiles[i], None if priors is None else priors[i])
        return out


def _soft(t: np.ndarray, thr: float, scale: float) -> np.ndarray:
    """Logistic soft threshold so classical output is a probability like the UNet's."""
    z = (t - thr) / max(scale, 1e-6)
    return (1.0 / (1.0 + np.exp(-np.clip(z * 2.0, -30, 30)))).astype(np.float32)
