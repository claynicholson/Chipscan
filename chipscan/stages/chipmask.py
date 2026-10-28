"""Where the chip is: skip everything else.

A wafer scan is mostly not chip: black fill from straightening, the stage around the wafer,
the bare rim of the wafer outside the dies.  Nothing there needs a network, a twin search or a
reviewer.  The chip area is found on a coarse level of the pyramid (a few thousand px across):
patterned pixels (local contrast), bridged and hole-filled so the flat ground plane *between*
structures stays in, then grown by a safety margin.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from chipscan.io.readers import ImageSource


class ChipMask:
    """Boolean chip area on a coarse level; `any_in(x0, y0, x1, y1)` asks in level-0 px."""

    def __init__(self, mask: np.ndarray, scale: int):
        self.mask, self.scale = mask, scale
        self._sum = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())

    def any_in(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        s = self.scale
        h, w = self.mask.shape
        a0, b0 = int(np.clip(y0 // s, 0, h)), int(np.clip(x0 // s, 0, w))
        a1, b1 = int(np.clip(-(-y1 // s), 0, h)), int(np.clip(-(-x1 // s), 0, w))
        if a1 <= a0 or b1 <= b0:
            return False
        c = self._sum
        return bool(c[a1, b1] - c[a0, b1] - c[a1, b0] + c[a0, b0] > 0)

    def full(self) -> bool:
        return bool(self.mask.all())


def chip_mask(source: ImageSource, max_px: int = 4096, margin_um: float = 400.0, pixel_size_um: float | None = None) -> ChipMask:
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > max_px:
        level += 1
    s = 2**level
    img = source.read_level(level).astype(np.float32) / 255.0
    if img.ndim == 3:
        img = img.mean(-1)
    valid = img > 0.01  # straightening fill / zero padding
    mean = ndi.uniform_filter(img, 5)
    std = np.sqrt(np.maximum(0.0, ndi.uniform_filter(img * img, 5) - mean * mean))
    patterned = (std > 0.025) & ndi.binary_erosion(valid, iterations=3)
    # isolated specks of contrast on a bare rim (dust, chips in the wafer edge) are not chip area
    lab, n = ndi.label(ndi.binary_dilation(patterned, iterations=2))
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= max(64, 0.0005 * patterned.size)
        patterned = keep[lab] & patterned
    px = (pixel_size_um or getattr(source, "pixel_size_um", None) or 1.0) * s
    bridge = max(2, int(round(150.0 / px)))  # flat ground plane between structures stays in
    m = ndi.binary_closing(patterned, iterations=bridge)
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_dilation(m, iterations=max(1, int(round(margin_um / px)))) & valid
    if m.mean() < 0.02:  # found nothing sensible (an odd image): scan everything
        m = valid if valid.mean() > 0.02 else np.ones_like(valid)
    return ChipMask(m, s)
