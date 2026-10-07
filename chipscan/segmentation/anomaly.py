"""Design-free anomaly segmentation: image in, anomaly outline out.

No GDS, no registration, no assumptions about what the structure is.  Training is
self-supervised in the DRAEM / CutPaste family, made design-agnostic:

* tiles are drawn from any number of *scenes* (real microscope mosaics, PCB scans,
  synthetic chips), each assumed to be mostly defect-free;
* synthetic anomalies are injected into the tile and the network learns to outline
  them.  Anomaly types are chosen to cover what a defect can look like without
  knowing the design: foreign structure (a patch pasted from elsewhere, possibly
  another scene), bridges/cuts drawn in the tile's own bright and dark tones,
  scratches, local discoloration, blur spots, and erasures (a region replaced by
  the tile's dominant tone);
* heavy appearance randomisation (polarity, gamma, contrast, blur, noise, scale,
  rotation) so the model does not learn one microscope or one chip.

Real defects that happen to be in the training tiles are unlabelled and rare; the
network learns "looks unlike its surroundings and unlike regular structure" rather
than "equals the design".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chipscan.geometry import BBox
from chipscan.io.readers import ImageSource


@dataclass
class AnomalyInjection:
    p_anomaly_tile: float = 0.8
    max_per_tile: int = 3
    min_size_px: float = 3.0
    max_size_px: float = 40.0
    p_types: dict[str, float] = field(
        # "paste" (foreign structure) teaches the model that unfamiliar structure is anomalous, which hurts
        # generalisation to new designs; keep it rare and prefer local, defect-like perturbations
        default_factory=lambda: {"paste": 0.05, "bar": 0.25, "blob": 0.25, "scratch": 0.12, "shade": 0.1, "blur": 0.05, "erase": 0.18}
    )
    edge_softness_px: float = 0.8
    p_text: float = 0.35
    """Probability of drawing unlabelled text (chip labels, markings) on a tile: text is normal, not an anomaly."""


def _perlin_like(h: int, w: int, rng: np.random.Generator, scale: int = 16) -> np.ndarray:
    """Smooth random field in [0,1] (sum of upsampled noise octaves)."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    s = scale
    while s >= 2:
        n = rng.uniform(size=(max(2, h // s), max(2, w // s))).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        s //= 2
    out /= max(tot, 1e-6)
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


class AnomalyTiles(Dataset):
    def __init__(
        self,
        scenes: list[ImageSource],
        tile_px: int = 256,
        n_samples: int = 2000,
        injection: AnomalyInjection | None = None,
        augment: bool = True,
        seed: int = 0,
        min_std: float = 0.01,
        regions: list[BBox | None] | None = None,
    ):
        if not scenes:
            raise ValueError("need at least one scene")
        self.scenes = scenes
        self.tile_px, self.n = tile_px, n_samples
        self.inj = injection or AnomalyInjection()
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.min_std = min_std
        self.regions = regions or [None] * len(scenes)

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------
    def _read(self, k: int, size: int) -> np.ndarray:
        sc = self.scenes[k]
        reg = self.regions[k] or sc.bbox_px
        for _ in range(30):
            x = int(self.rng.uniform(reg.x0, max(reg.x1 - size, reg.x0)))
            y = int(self.rng.uniform(reg.y0, max(reg.y1 - size, reg.y0)))
            t = sc.read_region(x, y, x + size, y + size, level=0).astype(np.float32) / 255.0
            if t.std() > self.min_std or self.rng.uniform() < 0.05:
                return t
        return t

    def _draw_tile(self) -> tuple[np.ndarray, int]:
        k = int(self.rng.integers(0, len(self.scenes)))
        # scale jitter: read a larger/smaller window and resize to tile_px -> scale invariance
        f = float(self.rng.uniform(0.6, 1.6)) if self.augment else 1.0
        size = max(32, int(round(self.tile_px * f)))
        t = self._read(k, size)
        if size != self.tile_px:
            t = cv2.resize(t, (self.tile_px, self.tile_px), interpolation=cv2.INTER_AREA if f > 1 else cv2.INTER_LINEAR)
        return t, k

    # ------------------------------------------------------------------
    def _shape(self, h: int, w: int, cx: float, cy: float, kind: str) -> np.ndarray:
        r = float(self.rng.uniform(self.inj.min_size_px, self.inj.max_size_px))
        m = np.zeros((h, w), np.uint8)
        if kind == "bar" or kind == "scratch":
            # bars are short and irregular on purpose: long uniform bars look like designed traces
            length = r * float(self.rng.uniform(1.2, 3.5)) if kind == "bar" else float(self.rng.uniform(30, w))
            width = max(1.0, r * float(self.rng.uniform(0.2, 0.7))) if kind == "bar" else float(self.rng.uniform(1.0, 2.5))
            pts = cv2.boxPoints(((cx, cy), (length, width), float(self.rng.uniform(0, 180)))).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
            if kind == "bar" and width >= 3:
                rough = (_perlin_like(h, w, self.rng, scale=6) > 0.55).astype(np.uint8)
                m = m & (1 - (rough & (cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        else:
            # organic blob: threshold a smooth field inside a disk
            rr = int(r)
            x0, y0 = max(int(cx) - rr, 0), max(int(cy) - rr, 0)
            x1, y1 = min(int(cx) + rr, w), min(int(cy) + rr, h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                return m
            field_ = _perlin_like(y1 - y0, x1 - x0, self.rng, scale=max(4, rr // 2))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
            thr = float(self.rng.uniform(0.35, 0.6))
            m[y0:y1, x0:x1] = (disk & (field_ > thr)).astype(np.uint8)
            if m.sum() < 4:
                cv2.circle(m, (int(cx), int(cy)), max(2, rr // 2), 1, -1)
        return m

    _FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX)
    _CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789-"

    def draw_text(self, img: np.ndarray) -> np.ndarray:
        """Etched/printed labels are designed structure: draw some, unlabelled, so the model treats text as normal."""
        h, w = img.shape
        out = img.copy()
        n = int(self.rng.integers(1, 4))
        for _ in range(n):
            txt = "".join(self.rng.choice(list(self._CHARS), size=int(self.rng.integers(1, 9))))
            font = int(self.rng.choice(self._FONTS))
            scale = float(self.rng.uniform(0.5, 7.0))  # real chip labels can be well over 100 px tall
            thick = int(self.rng.integers(1, 3 + int(scale)))
            (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
            layer8 = np.zeros((h, w), np.uint8)
            lo_x = -tw * 0.3
            x = int(self.rng.uniform(lo_x, max(lo_x + 1.0, w - tw * 0.7)))
            y = int(self.rng.uniform(min(th, h - 1), h))
            cv2.putText(layer8, txt, (x, y), font, scale, 255, thick, cv2.LINE_AA)
            layer = layer8.astype(np.float32) / 255.0
            k = int(self.rng.integers(0, 4))
            if k:
                layer = np.rot90(layer, k)
                if layer.shape != (h, w):
                    layer = cv2.resize(layer, (w, h))
            if self.rng.uniform() < 0.5:  # hollow / outlined letters, as etched labels often look
                er = cv2.erode((layer > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
                layer = layer * (1 - er) if er.any() else layer
            local = float(np.median(img[layer > 0.5])) if (layer > 0.5).any() else float(img.mean())
            delta = float(self.rng.uniform(0.08, 0.35)) * (1 if self.rng.uniform() < 0.5 else -1)
            tone = float(np.clip(local + delta, 0, 1))
            out = out * (1 - layer) + tone * layer
        return np.clip(out, 0, 1).astype(np.float32)

    def inject(self, img: np.ndarray, scene_idx: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape
        target = np.zeros((h, w), np.float32)
        if self.rng.uniform() < self.inj.p_text:
            img = self.draw_text(img)
        out = img.copy()
        if self.rng.uniform() > self.inj.p_anomaly_tile:
            return out, target
        lo, hi = float(np.percentile(img, 5)), float(np.percentile(img, 95))
        if hi - lo < 0.02:
            hi = lo + 0.2
        kinds, probs = zip(*self.inj.p_types.items(), strict=True)
        probs = np.array(probs) / sum(probs)
        n = int(self.rng.integers(1, self.inj.max_per_tile + 1))
        for _ in range(n):
            kind = str(self.rng.choice(kinds, p=probs))
            cx, cy = float(self.rng.uniform(4, w - 4)), float(self.rng.uniform(4, h - 4))
            m = self._shape(h, w, cx, cy, "bar" if kind == "bar" else ("scratch" if kind == "scratch" else "blob"))
            if m.sum() < 3:
                continue
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), self.inj.edge_softness_px) if self.inj.edge_softness_px > 0 else m.astype(np.float32)
            soft = np.clip(soft * 1.3, 0, 1)
            # local background tone under the anomaly, so painted tones always contrast with it
            local = float(np.median(img[m > 0])) if (m > 0).any() else float(img.mean())
            if kind == "paste":
                # foreign structure: a patch from elsewhere (same or another scene), randomly flipped/rotated
                src_k = scene_idx  # same scene only: the pasted structure is at least from the same product
                patch = self._read(src_k, h) if self.scenes[src_k].shape[0] >= h else img
                if patch.shape != img.shape:
                    patch = cv2.resize(patch, (w, h))
                patch = np.rot90(patch, int(self.rng.integers(0, 4)))
                if self.rng.uniform() < 0.5:
                    patch = patch[:, ::-1]
                # match the local tone so it is the structure, not the brightness, that is foreign
                patch = np.clip(patch - patch.mean() + img.mean() + self.rng.normal(0, 0.05), 0, 1)
                tex = patch
            elif kind in ("bar", "blob", "scratch"):
                # contrast of at least 0.12 against the local tone, in a random direction, bounded to the image's
                # own dynamic range when possible (looks like the other material) but never invisible
                delta = float(self.rng.uniform(0.12, 0.45))
                if self.rng.uniform() < 0.5:
                    tone = local + delta if local + delta <= 1.0 else local - delta
                else:
                    tone = local - delta if local - delta >= 0.0 else local + delta
                tone = float(np.clip(tone, 0, 1))
                tex = np.full_like(img, tone) + self.rng.normal(0, 0.02 if kind != "scratch" else 0.0, img.shape).astype(np.float32)
            elif kind == "shade":
                tex = np.clip(img * float(self.rng.uniform(0.6, 1.4)) + float(self.rng.uniform(-0.15, 0.15)), 0, 1)
            elif kind == "blur":
                tex = cv2.GaussianBlur(img, (0, 0), float(self.rng.uniform(2.0, 5.0)))
            else:  # erase: replace by the dominant tone
                hist, edges = np.histogram(img, bins=32, range=(0, 1))
                tone = float(edges[int(np.argmax(hist))] + 1 / 64)
                if abs(tone - local) < 0.08:  # erasing with the same tone is invisible: use the other extreme
                    tone = hi if abs(hi - local) > abs(lo - local) else lo
                tex = np.full_like(img, tone) + self.rng.normal(0, 0.015, img.shape).astype(np.float32)
            changed = np.abs(tex - img) > 0.04
            if (changed & (m > 0)).sum() < 3:
                continue  # invisible change; do not label it
            out = out * (1 - soft) + tex * soft
            label = (m > 0) & changed  # label only what is actually visible
            target = np.maximum(target, label.astype(np.float32))
        return np.clip(out, 0, 1).astype(np.float32), target

    # ------------------------------------------------------------------
    def __getitem__(self, i: int):
        img, k = self._draw_tile()
        img, target = self.inject(img, k)
        if self.augment:
            r = int(self.rng.integers(0, 4))
            img, target = np.rot90(img, r), np.rot90(target, r)
            if self.rng.uniform() < 0.5:
                img, target = img[:, ::-1], target[:, ::-1]
            g = float(self.rng.uniform(0.6, 1.6))  # gamma
            img = np.clip(img, 0, 1) ** g
            img = np.clip(img * float(self.rng.uniform(0.7, 1.3)) + float(self.rng.uniform(-0.15, 0.15)), 0, 1)
            if self.rng.uniform() < 0.3:
                img = np.clip(img + self.rng.normal(0, float(self.rng.uniform(0.005, 0.05)), img.shape), 0, 1)
            if self.rng.uniform() < 0.25:
                img = cv2.GaussianBlur(np.ascontiguousarray(img, dtype=np.float32), (0, 0), float(self.rng.uniform(0.5, 1.5)))
            if self.rng.uniform() < 0.3:
                img = 1.0 - img  # polarity
            if self.rng.uniform() < 0.2:  # jpeg-like blockiness / mosaic brightness steps
                step = int(self.rng.integers(32, 128))
                yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
                img = np.clip(img + self.rng.uniform(-0.03, 0.03, (img.shape[0] // step + 1, img.shape[1] // step + 1))[yy // step, xx // step], 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None]
        y = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))[None]
        return x, y


# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Inference wrapper: tiles any ImageSource and returns an anomaly probability map."""

    def __init__(self, model_path: str | Path, device: str = "auto", batch_size: int = 8):
        from chipscan.segmentation.unet import UNet, pick_device

        self.device = pick_device(device)
        self.model = UNet.load(model_path, self.device).to(self.device).eval()
        self.batch_size = batch_size
        self.name = f"anomaly:{Path(model_path).name}"

    @torch.no_grad()
    def predict_tiles(self, tiles: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F

        out = np.empty(tiles.shape, np.float32)
        for i in range(0, len(tiles), self.batch_size):
            x = torch.from_numpy(tiles[i : i + self.batch_size]).float()[:, None]
            h, w = x.shape[-2:]
            m = 2**self.model.depth
            ph, pw = (-h) % m, (-w) % m
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            y = torch.sigmoid(self.model(x.to(self.device))).cpu().numpy()[:, 0, :h, :w]
            out[i : i + len(y)] = y
        return out

    def run(self, source: ImageSource, tile_px: int = 512, margin_px: int = 48, store=None, progress: bool = True) -> np.ndarray | None:
        """Write the anomaly map (uint8 0..255) into `store` (zarr array shaped like the image) or return it."""
        from rich.progress import Progress

        from chipscan.tiling import make_tiles

        tiles = make_tiles(source.shape, tile_px, margin_px)
        full = None if store is not None else np.zeros(source.shape, np.uint8)
        it = tiles
        prog = Progress() if progress else None
        if prog:
            prog.start()
            task = prog.add_task(f"anomaly ({self.name})", total=len(tiles))
        for t in it:
            x0, y0, x1, y1 = t.read.to_int()
            img = source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0
            p = self.predict_tiles(img[None])[0][t.core_slice_in_read]
            cx0, cy0, cx1, cy1 = t.core.to_int()
            arr = np.rint(p * 255).astype(np.uint8)
            if store is not None:
                store[cy0:cy1, cx0:cx1] = arr
            else:
                full[cy0:cy1, cx0:cx1] = arr
            if prog:
                prog.advance(task)
        if prog:
            prog.stop()
        return full


def anomalies_from_map(amap, threshold: float = 0.5, min_area_px: int = 6, pixel_size_um: float | None = None, chunk: int = 4096) -> list[dict]:
    """Connected components of a (possibly zarr-backed) anomaly map -> outline records in px (and um if known)."""
    from scipy import ndimage as ndi

    H, W = amap.shape
    thr = int(round(threshold * 255))
    recs: list[dict] = []
    nid = 0
    pad = 32
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y0 + chunk + pad, H), min(x0 + chunk + pad, W)
            blk = np.asarray(amap[ya:yb, xa:xb])
            m = blk > thr
            if not m.any():
                continue
            lab, n = ndi.label(m)
            objs = ndi.find_objects(lab)
            for i, sl in enumerate(objs, start=1):
                if sl is None:
                    continue
                comp = lab[sl] == i
                area = int(comp.sum())
                if area < min_area_px:
                    continue
                cy, cx = ndi.center_of_mass(comp)
                gy, gx = cy + sl[0].start + ya, cx + sl[1].start + xa
                # keep only components whose centroid lies in this chunk's core (avoid double counting in the pad)
                if not (y0 <= gy < y0 + chunk and x0 <= gx < x0 + chunk):
                    continue
                cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [(c.reshape(-1, 2).astype(float) + [sl[1].start + xa + 0.5, sl[0].start + ya + 0.5]).round(2).tolist() for c in cs if len(c) >= 3]
                score = float(blk[sl][comp].mean() / 255.0)
                bb = {"x0": float(sl[1].start + xa), "y0": float(sl[0].start + ya), "x1": float(sl[1].stop + xa), "y1": float(sl[0].stop + ya)}
                rec = {"id": nid, "kind": "anomaly", "class": "anomaly", "area_px": area, "score": score, "bbox_px": bb, "centroid_px": [float(gx), float(gy)], "polygons_px": polys}
                if pixel_size_um:
                    rec["area_um2"] = area * pixel_size_um**2
                    rec["centroid_um"] = [float(gx) * pixel_size_um, float(gy) * pixel_size_um]
                    rec["bbox_um"] = {k: v * pixel_size_um for k, v in bb.items()}
                else:
                    rec["area_um2"] = float(area)
                    rec["centroid_um"] = [float(gx), float(gy)]
                    rec["bbox_um"] = dict(bb)
                recs.append(rec)
                nid += 1
    return recs


def write_anomaly_outputs(out_dir: Path, name: str, recs: list[dict], source: ImageSource, amap, pixel_size_um: float | None, overlay_max_px: int = 3000) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = getattr(source, "path", None)
    (out_dir / f"{name}.anomalies.json").write_text(
        json.dumps({"summary": {"n": len(recs), "pixel_size_um": pixel_size_um, "image": str(src_path) if src_path else None}, "defects": recs}, indent=2)
    )
    lines = ["id,area_px,area_um2,score,cx_px,cy_px"] + [f"{r['id']},{r['area_px']},{r['area_um2']:.2f},{r['score']:.3f},{r['centroid_px'][0]:.1f},{r['centroid_px'][1]:.1f}" for r in recs]
    (out_dir / f"{name}.anomalies.csv").write_text("\n".join(lines) + "\n")
    # overlay
    level = 0
    while level + 1 < source.num_levels and max(source.level_shape(level)) > overlay_max_px:
        level += 1
    s = 2**level
    img = source.read_level(level)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r in recs:
        b = r["bbox_px"]
        x0, y0, x1, y1 = (int(v / s) for v in (b["x0"], b["y0"], b["x1"], b["y1"]))
        cv2.rectangle(rgb, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 0, 255), 1)
    p = out_dir / f"{name}.anomaly_overlay.png"
    cv2.imwrite(str(p), rgb)
    return {"json": out_dir / f"{name}.anomalies.json", "csv": out_dir / f"{name}.anomalies.csv", "overlay": p}
