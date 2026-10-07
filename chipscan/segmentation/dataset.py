"""Self-labelled training data: the registered design *is* the label.

Random tiles are drawn from the image; the label is the design rendered through
the warp field.  A band around design edges is excluded from the loss so that
sub-pixel registration error is not learned as truth.  Real defects are rare
enough that they act as label noise the network ignores, which is the point:
the network learns the design, not the defects.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from chipscan.design import Design
from chipscan.detection.compare import tolerance_band
from chipscan.geometry import BBox, WarpField
from chipscan.io.readers import ImageSource


class DesignLabeledTiles(Dataset):
    def __init__(
        self,
        source: ImageSource,
        design: Design,
        warp: WarpField,
        tile_px: int = 256,
        n_samples: int = 2000,
        ignore_px: int = 2,
        use_prior: bool = False,
        augment: bool = True,
        seed: int = 0,
        region_px: BBox | None = None,
        min_content: float = 0.01,
    ):
        self.src, self.design, self.warp = source, design, warp
        self.tile_px, self.n, self.ignore_px = tile_px, n_samples, ignore_px
        self.use_prior, self.augment = use_prior, augment
        self.rng = np.random.default_rng(seed)
        ov = warp.affine.apply_bbox(design.bbox).intersection(source.bbox_px)
        if ov is None:
            raise ValueError("design does not overlap image")
        self.region = ov if region_px is None else ov.intersection(region_px) or ov
        self.min_content = min_content

    def __len__(self) -> int:
        return self.n

    def _draw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(20):
            x = int(self.rng.uniform(self.region.x0, max(self.region.x1 - self.tile_px, self.region.x0)))
            y = int(self.rng.uniform(self.region.y0, max(self.region.y1 - self.tile_px, self.region.y0)))
            bb = BBox(x, y, x + self.tile_px, y + self.tile_px)
            img = self.src.read_region(x, y, x + self.tile_px, y + self.tile_px, level=0)
            lab = self.design.render(self.warp.local_affine(bb), bb)
            frac = lab.mean()
            if self.min_content <= frac <= 1 - self.min_content or self.rng.uniform() < 0.1:
                break
        ign = tolerance_band(lab, self.ignore_px)
        return img, lab, ign

    def __getitem__(self, i: int):
        img, lab, ign = self._draw()
        x = img.astype(np.float32) / 255.0
        if self.augment:
            k = int(self.rng.integers(0, 4))
            x, lab, ign = np.rot90(x, k), np.rot90(lab, k), np.rot90(ign, k)
            if self.rng.uniform() < 0.5:
                x, lab, ign = x[:, ::-1], lab[:, ::-1], ign[:, ::-1]
            # photometric jitter: mosaics have illumination drift and colour shifts between tiles
            g = self.rng.uniform(0.8, 1.2)
            b = self.rng.uniform(-0.08, 0.08)
            x = np.clip(x * g + b, 0, 1)
            if self.rng.uniform() < 0.3:
                x = np.clip(x + self.rng.normal(0, 0.02, x.shape), 0, 1)
        chans = [x]
        if self.use_prior:
            chans.append(lab.astype(np.float32))
        xt = torch.from_numpy(np.ascontiguousarray(np.stack(chans))).float()
        yt = torch.from_numpy(np.ascontiguousarray(lab)).float()[None]
        wt = torch.from_numpy(np.ascontiguousarray(~ign)).float()[None]
        return xt, yt, wt
