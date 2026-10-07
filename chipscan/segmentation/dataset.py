"""Training data for the inspection UNet: self-labelled from the design, with
design-aware synthetic defect injection.

Why not simply "image -> design mask"?  That works for a plain segmenter, but it
cannot use the design as an *input*: if the rendered design is both an input
channel and the label, the network copies the input and never looks at the image.

So each sample is built as follows:

1. Draw a tile from a registered scene (image + design + warp).
2. Render the design for that tile (`prior`); optionally jitter it by a pixel so
   registration slop is learned as normal.
3. Inject synthetic defects *into the image*, guided by the design: blobs of
   metal texture pasted where the design has gaps (shorts, particles,
   protrusions) and blobs of substrate texture pasted on metal (opens, voids,
   notches).  Textures are sampled from the tile itself, so illumination drift,
   colour shifts and mosaic steps are reproduced for free.
4. Targets: ``metal`` (design, with the injected changes applied and a narrow
   ignore band at design edges), ``extra`` and ``missing`` (the injected regions).

The network therefore learns "where does the image disagree with the design",
which is the inspection task itself, without a single hand label.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chipscan.design import Design
from chipscan.detection.compare import tolerance_band
from chipscan.geometry import BBox, WarpField
from chipscan.io.readers import ImageSource


@dataclass
class Scene:
    """One registered image/design pair."""

    source: ImageSource
    design: Design
    warp: WarpField
    pixel_size_um: float
    name: str = ""
    region_px: BBox | None = None  # restrict sampling (e.g. hold-out strip)

    @property
    def sample_region(self) -> BBox:
        ov = self.warp.affine.apply_bbox(self.design.bbox).intersection(self.source.bbox_px)
        if ov is None:
            raise ValueError(f"scene {self.name}: design does not overlap image")
        if self.region_px is not None:
            r = ov.intersection(self.region_px)
            return r if r is not None else ov
        return ov


@dataclass
class InjectionConfig:
    enabled: bool = True
    p_defect_tile: float = 0.85
    """Probability a tile receives any injected defect."""
    max_per_tile: int = 4
    min_radius_px: float = 1.5
    max_radius_px: float = 12.0
    p_bar: float = 0.35
    """Fraction of blobs that are elongated bars (bridges/cuts) rather than round blobs."""
    edge_softness_px: float = 0.7
    texture_noise: float = 1.0
    """Scale of noise added to pasted texture relative to the tile's own class std."""


class InspectionTiles(Dataset):
    def __init__(
        self,
        scenes: list[Scene],
        tile_px: int = 256,
        n_samples: int = 2000,
        ignore_px: int = 1,
        use_prior: bool = True,
        prior_jitter_px: float = 1.0,
        augment: bool = True,
        injection: InjectionConfig | None = None,
        seed: int = 0,
        min_content: float = 0.02,
    ):
        if not scenes:
            raise ValueError("need at least one scene")
        self.scenes = scenes
        self.tile_px, self.n, self.ignore_px = tile_px, n_samples, ignore_px
        self.use_prior, self.prior_jitter_px, self.augment = use_prior, prior_jitter_px, augment
        self.inj = injection or InjectionConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.min_content = min_content
        self._regions = [s.sample_region for s in scenes]

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------
    def _draw_tile(self) -> tuple[np.ndarray, np.ndarray, Scene]:
        k = int(self.rng.integers(0, len(self.scenes)))
        sc, reg = self.scenes[k], self._regions[k]
        for _ in range(20):
            x = int(self.rng.uniform(reg.x0, max(reg.x1 - self.tile_px, reg.x0)))
            y = int(self.rng.uniform(reg.y0, max(reg.y1 - self.tile_px, reg.y0)))
            bb = BBox(x, y, x + self.tile_px, y + self.tile_px)
            img = sc.source.read_region(x, y, x + self.tile_px, y + self.tile_px, level=0)
            lab = sc.design.render(sc.warp.local_affine(bb), bb)
            frac = lab.mean()
            if self.min_content <= frac <= 1 - self.min_content or self.rng.uniform() < 0.15:
                break
        return img.astype(np.float32) / 255.0, lab.astype(np.uint8), sc

    # ------------------------------------------------------------------
    def _class_stats(self, img: np.ndarray, lab: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        far = ~tolerance_band(lab, 2)
        m, s = img[(lab > 0) & far], img[(lab == 0) & far]
        if m.size < 20:
            m = img[lab > 0] if (lab > 0).any() else img
        if s.size < 20:
            s = img[lab == 0] if (lab == 0).any() else img
        return (float(m.mean()), float(m.std()) + 1e-3), (float(s.mean()), float(s.std()) + 1e-3)

    def _blob(self, h: int, w: int, cx: float, cy: float) -> np.ndarray:
        """Soft-edged random blob mask (float 0..1)."""
        r = float(self.rng.uniform(self.inj.min_radius_px, self.inj.max_radius_px))
        m = np.zeros((h, w), dtype=np.uint8)
        if self.rng.uniform() < self.inj.p_bar:
            length = r * float(self.rng.uniform(2.0, 6.0))
            width = max(1.0, r * float(self.rng.uniform(0.25, 0.8)))
            ang = float(self.rng.uniform(0, 180))
            box = cv2.boxPoints(((cx, cy), (length, width), ang)).astype(np.int32)
            cv2.fillPoly(m, [box], 1)
        else:
            axes = (int(max(1, r)), int(max(1, r * float(self.rng.uniform(0.5, 1.0)))))
            cv2.ellipse(m, (int(cx), int(cy)), axes, float(self.rng.uniform(0, 180)), 0, 360, 1, -1)
            # roughen the outline a little
            if r > 3 and self.rng.uniform() < 0.5:
                noise = (self.rng.uniform(size=(h, w)) < 0.5).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) | (m & noise)
        f = m.astype(np.float32)
        if self.inj.edge_softness_px > 0:
            f = cv2.GaussianBlur(f, (0, 0), self.inj.edge_softness_px)
        return f

    def inject(self, img: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (image_with_defects, metal_after, extra_mask, missing_mask)."""
        h, w = lab.shape
        extra = np.zeros((h, w), dtype=np.float32)
        missing = np.zeros((h, w), dtype=np.float32)
        img = img.copy()
        if not self.inj.enabled or self.rng.uniform() > self.inj.p_defect_tile:
            return img, lab.copy(), extra, missing
        (mu_m, sd_m), (mu_s, sd_s) = self._class_stats(img, lab)
        n = int(self.rng.integers(1, self.inj.max_per_tile + 1))
        metal_after = lab.astype(np.float32)
        for _ in range(n):
            want_extra = self.rng.uniform() < 0.5
            cand = np.argwhere((lab == 0) if want_extra else (lab > 0))
            if len(cand) == 0:
                continue
            cy, cx = cand[int(self.rng.integers(0, len(cand)))]
            soft = self._blob(h, w, float(cx), float(cy))
            # decide the label first (hard mask, restricted to where the design allows the change),
            # then blend the image with a softened copy of *that* mask so every changed pixel is labelled
            hard = (soft > 0.5) & ((lab == 0) if want_extra else (lab > 0))
            if not hard.any():
                continue
            blend = hard.astype(np.float32)
            if self.inj.edge_softness_px > 0:
                blend = np.clip(cv2.GaussianBlur(blend, (0, 0), self.inj.edge_softness_px) * 1.2, 0, 1)
            if want_extra:
                tex = mu_m + self.rng.normal(0, sd_m * self.inj.texture_noise, (h, w)).astype(np.float32)
                extra = np.maximum(extra, hard.astype(np.float32))
            else:
                tex = mu_s + self.rng.normal(0, sd_s * self.inj.texture_noise, (h, w)).astype(np.float32)
                missing = np.maximum(missing, hard.astype(np.float32))
            img = img * (1 - blend) + tex * blend
            metal_after = np.where(hard, 1.0 if want_extra else 0.0, metal_after)
        img = np.clip(img, 0, 1).astype(np.float32)
        return img, (metal_after > 0.5).astype(np.uint8), (extra > 0.5).astype(np.float32), (missing > 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    def _jitter(self, prior: np.ndarray) -> np.ndarray:
        if self.prior_jitter_px <= 0:
            return prior
        dx, dy = self.rng.uniform(-self.prior_jitter_px, self.prior_jitter_px, 2)
        m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return cv2.warpAffine(prior.astype(np.float32), m, (prior.shape[1], prior.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def __getitem__(self, i: int):
        img, lab, _ = self._draw_tile()
        img, metal, extra, missing = self.inject(img, lab)
        prior = self._jitter(lab).astype(np.float32)
        ign = tolerance_band(metal, self.ignore_px)
        if self.augment:
            k = int(self.rng.integers(0, 4))
            arrs = [np.rot90(a, k) for a in (img, prior, metal, extra, missing, ign)]
            if self.rng.uniform() < 0.5:
                arrs = [a[:, ::-1] for a in arrs]
            img, prior, metal, extra, missing, ign = arrs
            g = self.rng.uniform(0.75, 1.25)
            b = self.rng.uniform(-0.1, 0.1)
            img = np.clip(img * g + b, 0, 1)
            if self.rng.uniform() < 0.3:
                img = np.clip(img + self.rng.normal(0, self.rng.uniform(0.005, 0.04), img.shape), 0, 1)
            if self.rng.uniform() < 0.2:  # occasional blur (focus variation across a wafer)
                img = cv2.GaussianBlur(np.ascontiguousarray(img, dtype=np.float32), (0, 0), float(self.rng.uniform(0.5, 1.5)))
            if self.rng.uniform() < 0.15:  # polarity flip: some products/illuminations make metal dark
                img = 1.0 - img
        chans = [img] + ([prior] if self.use_prior else [])
        x = torch.from_numpy(np.ascontiguousarray(np.stack(chans), dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(np.stack([metal.astype(np.float32), extra, missing]), dtype=np.float32))
        wgt = torch.from_numpy(np.ascontiguousarray(~ign, dtype=np.float32))[None]
        return x, y, wgt


def split_scene(scene: Scene, val_frac: float = 0.1) -> tuple[Scene, Scene]:
    """Spatial split: the right-most strip of the scene is held out for validation."""
    r = scene.sample_region
    cut = r.x0 + (1 - val_frac) * r.width
    tr = Scene(scene.source, scene.design, scene.warp, scene.pixel_size_um, scene.name + ":train", BBox(r.x0, r.y0, cut, r.y1))
    va = Scene(scene.source, scene.design, scene.warp, scene.pixel_size_um, scene.name + ":val", BBox(cut, r.y0, r.x1, r.y1))
    return tr, va
