"""The three networks of the pipeline, their training sets, and tiled inference.

* ``speck``  - binary UNet: dirt (dust, particles, fibres, residue, probe scrub marks).
* ``trace``  - binary UNet: shape errors of traces (shorts, opens, bites, protrusions,
  width changes, missing stretches).
* ``layout`` - 4-class UNet: background / trace / component / text, used when no GDS
  is given and for weighting every finding by what it sits next to.

All three train on real tiles from any number of scans, self-supervised: speck and trace
nets see the same painted tiles (see ``inject.py``) but each is only taught its own category;
the layout net learns from pseudo-labels derived from tone and shape (plus painted text).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from chipscan.io.readers import ImageSource
from chipscan.segmentation.unet import UNet, pick_device
from chipscan.stages.inject import Tones, compose, paint_text

LAYOUT_CLASSES = ("background", "trace", "component", "text")
IGNORE = 255


# ----------------------------------------------------------------------------- base tiles
class ScanTiles(Dataset):
    def __init__(self, scenes: list[ImageSource], tile_px: int = 256, n_samples: int = 2000, seed: int = 0, augment: bool = True, min_std: float = 0.012):
        if not scenes:
            raise ValueError("need at least one scene")
        self.scenes, self.tile_px, self.n, self.augment, self.min_std = scenes, tile_px, n_samples, augment, min_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    hard: list | None = None  # per scene: (n, 2) array of hard-negative centres, or None
    p_hard: float = 0.0

    def read_tile(self) -> np.ndarray:
        k = int(self.rng.integers(len(self.scenes)))
        sc = self.scenes[k]
        f = float(self.rng.uniform(0.8, 1.25)) if self.augment else 1.0
        size = max(32, int(round(self.tile_px * f)))
        H, W = sc.shape
        t = None
        if self.hard and self.rng.uniform() < self.p_hard:
            ks = [i for i, h in enumerate(self.hard) if h is not None and len(h)]
            if ks:
                k = int(self.rng.choice(ks))
                sc = self.scenes[k]
                H, W = sc.shape
                cx, cy = self.hard[k][int(self.rng.integers(len(self.hard[k])))]
                j = size // 3
                x = int(np.clip(cx - size // 2 + self.rng.integers(-j, j + 1), 0, max(0, W - size)))
                y = int(np.clip(cy - size // 2 + self.rng.integers(-j, j + 1), 0, max(0, H - size)))
                t = sc.read_region(x, y, x + size, y + size).astype(np.float32) / 255.0
                if size != self.tile_px:
                    t = cv2.resize(t, (self.tile_px, self.tile_px), interpolation=cv2.INTER_AREA if f > 1 else cv2.INTER_LINEAR)
                return t
        for _ in range(30):
            x, y = int(self.rng.uniform(0, max(1, W - size))), int(self.rng.uniform(0, max(1, H - size)))
            t = sc.read_region(x, y, x + size, y + size).astype(np.float32) / 255.0
            if t.std() > self.min_std or self.rng.uniform() < 0.05:
                break
        if size != self.tile_px:
            t = cv2.resize(t, (self.tile_px, self.tile_px), interpolation=cv2.INTER_AREA if f > 1 else cv2.INTER_LINEAR)
        return t

    def geometric(self, *arrs: np.ndarray) -> list[np.ndarray]:
        r = int(self.rng.integers(4))
        flip = self.rng.uniform() < 0.5
        out = []
        for a in arrs:
            a = np.rot90(a, r)
            out.append(np.ascontiguousarray(a[:, ::-1] if flip else a))
        return out

    def photometric(self, img: np.ndarray) -> np.ndarray:
        if not self.augment:
            return img
        img = np.clip(img, 0, 1) ** float(self.rng.uniform(0.7, 1.4))
        img = np.clip(img * float(self.rng.uniform(0.75, 1.3)) + float(self.rng.uniform(-0.12, 0.12)), 0, 1)
        if self.rng.uniform() < 0.3:
            img = np.clip(img + self.rng.normal(0, float(self.rng.uniform(0.004, 0.03)), img.shape), 0, 1)
        if self.rng.uniform() < 0.2:
            img = cv2.GaussianBlur(np.ascontiguousarray(img, np.float32), (0, 0), float(self.rng.uniform(0.4, 1.2)))
        if self.rng.uniform() < 0.1:
            img = 1.0 - img
        return img.astype(np.float32)


def real_dirt(img: np.ndarray, t: Tones | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Real dirt we can name without a model, and blobs we cannot judge.

    * dirt on pads: a bonding/probe pad is designed as one flat bright area, so anything dark
      in its interior (probe scrub marks, particles) is dirt.  These are *real* positives: the
      scans are full of them and a net trained only on painted dirt learns they are normal.
    * uncertain: small isolated blobs elsewhere (dirt, or a tiny designed feature) - excluded
      from the loss rather than taught as normal.
    """
    t = t or Tones(img)
    dirt = np.zeros(img.shape, bool)
    bl, n = ndi.label(t.bright_mask)
    for i, sl in enumerate(ndi.find_objects(bl), start=1):
        if sl is None:
            continue
        m = bl[sl] == i
        if m.sum() < 2500 or min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < 40:
            continue  # bond/probe pads only; junction pads and thin bright lines are not "pads"
        filled = ndi.binary_fill_holes(m)
        interior = ndi.binary_erosion(filled, iterations=10)
        if interior.sum() < 200:
            continue
        tone = float(np.median(img[sl][m & interior])) if (m & interior).any() else t.bright
        d = interior & (img[sl] < tone - max(0.06, 4 * t.noise))
        if d.sum() >= 6:
            lab, k = ndi.label(d)
            keep = np.zeros(k + 1, bool)
            for j, sj in enumerate(ndi.find_objects(lab), start=1):
                if sj is None:
                    continue
                cj = lab[sj] == j
                hh, ww = sj[0].stop - sj[0].start, sj[1].stop - sj[1].start
                # a designed frame/line is long, thin and fills its box; dirt is a compact irregular cluster
                straight = max(hh, ww) >= 4 * min(hh, ww) and cj.sum() >= 0.6 * hh * ww
                # stubs of the frame survive the erosion as thin marks on the interior's edge
                rim = (interior & ~ndi.binary_erosion(interior, iterations=3))[sj] & cj
                sliver = min(hh, ww) <= 3 and max(hh, ww) >= 8
                keep[j] = cj.sum() >= 6 and not straight and not sliver and rim.sum() < 0.5 * cj.sum()
            d = keep[lab]
            dirt[sl] |= ndi.binary_dilation(d, iterations=1) & filled
    unsure = np.zeros(img.shape, bool)
    struct = t.dark_mask | t.bright_mask
    lab, n = ndi.label(struct)
    # isolated = alone within 8 px; letters, arrays and devices always have neighbours
    grp, _ = ndi.label(ndi.binary_dilation(struct, iterations=8))
    per_grp = np.bincount(np.unique(np.stack([grp[struct], lab[struct]], 1), axis=0)[:, 0]) if struct.any() else np.zeros(1, int)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        g = grp[sl][m][0]
        if per_grp[g] > 1:
            continue
        area = int(m.sum())
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 3 <= area <= 400 and max(hh, ww) <= 30 and area / float(hh * ww) < 0.6:  # small and irregular
            unsure[sl] |= m
    unsure = ndi.binary_dilation(unsure, iterations=2) & ~dirt
    return dirt, unsure


class DefectTiles(ScanTiles):
    """Painted tiles; `target` is the category that is labelled ("speck" or "trace").

    Items are (image, target, weight): weight 0 marks pixels the loss must ignore."""

    def __init__(self, scenes, target: str, tile_px: int = 256, n_samples: int = 2000, seed: int = 0, augment: bool = True, p_clean: float = 0.2):
        super().__init__(scenes, tile_px, n_samples, seed, augment)
        assert target in ("speck", "trace")
        self.target, self.p_clean = target, p_clean

    def __getitem__(self, i):
        img = self.read_tile()
        rng = self.rng
        dirt, unsure = real_dirt(img) if self.target == "speck" else (None, None)
        clean = rng.uniform() < self.p_clean
        own = 0 if clean else int(rng.integers(1, 4))
        other = int(rng.poisson(1.0))
        counts = {
            "speck": own if self.target == "speck" else other,
            "trace": own if self.target == "trace" else other,
            "text": int(rng.uniform() < 0.45) + int(rng.uniform() < 0.15),
            "mark": int(rng.poisson(0.8)),
        }
        img, masks = compose(img, rng, counts)
        y = masks[self.target].astype(np.float32)
        w = np.ones_like(y)
        if dirt is not None:
            painted_other = np.zeros_like(dirt)
            for k, m in masks.items():
                if k != self.target:
                    painted_other |= m
            y[dirt & ~painted_other] = 1.0
            w[unsure & ~painted_other & (y < 0.5)] = 0.0
        img, y, w = self.geometric(img, y, w)
        img = self.photometric(img)
        return torch.from_numpy(img)[None], torch.from_numpy(y)[None], torch.from_numpy(w)[None]


# ----------------------------------------------------------------------------- layout pseudo-labels
def pseudo_layout(img: np.ndarray) -> np.ndarray:
    """Tone/shape rules -> {0 background, 1 trace, 2 component, 255 ignore}.

    Components are small, compact, rectangular bright features with their dark outline and
    the short link between a pad pair (Josephson junctions, test devices).  Large bright areas
    (pads) and every elongated dark structure (gaps, line edges) are traces.  Small irregular
    blobs are left unlabelled: they may be dirt, and the layout net must not learn dirt.
    """
    t = Tones(img)
    lab = np.zeros(img.shape, np.uint8)
    struct = t.dark_mask | t.bright_mask
    # bright compact rectangles -> components
    bl, n = ndi.label(t.bright_mask)
    comp = np.zeros(img.shape, bool)
    big_bright = np.zeros(img.shape, bool)
    for i, sl in enumerate(ndi.find_objects(bl), start=1):
        if sl is None:
            continue
        m = bl[sl] == i
        area = int(m.sum())
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if area > 700 or max(hh, ww) > 45:
            big_bright[sl] |= m
        elif area >= 12 and area / float(hh * ww) >= 0.55 and min(hh, ww) >= 3:
            comp[sl] |= m
    if comp.any():
        # outline + the link between pad pairs belong to the component
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        closed = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
        comp = comp | (grown & struct) | (closed & struct)
    # dark structure: elongated or large -> trace, small compact -> ignore (could be dirt)
    dl, n = ndi.label(t.dark_mask & ~comp)
    trace = big_bright.copy()
    ignore = np.zeros(img.shape, bool)
    for i, sl in enumerate(ndi.find_objects(dl), start=1):
        if sl is None:
            continue
        m = dl[sl] == i
        area = int(m.sum())
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == img.shape[0] or sl[1].stop == img.shape[1]
        if area >= 60 or max(hh, ww) >= 3 * max(1, min(hh, ww)) or touches:
            trace[sl] |= m
        else:
            ignore[sl] |= m
    # remaining small bright bits that are neither pad nor component
    rest = t.bright_mask & ~comp & ~big_bright
    ignore |= rest
    lab[trace] = 1
    lab[comp] = 2
    lab[ignore] = IGNORE
    # a thin band around structure is ambiguous at the pixel level
    edge = cv2.dilate(struct.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~struct
    lab[edge & (lab == 0)] = IGNORE
    return lab


class LayoutTiles(ScanTiles):
    def __init__(self, scenes, tile_px: int = 256, n_samples: int = 2000, seed: int = 0, augment: bool = True, p_text: float = 0.5):
        super().__init__(scenes, tile_px, n_samples, seed, augment)
        self.p_text = p_text

    def __getitem__(self, i):
        img = self.read_tile()
        lab = pseudo_layout(img)
        if self.rng.uniform() < self.p_text:
            t = Tones(img)
            for _ in range(int(self.rng.integers(1, 3))):
                img, vis = paint_text(img, self.rng, t)
                tm = cv2.dilate(vis.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                lab[tm] = 3
        img, lab = self.geometric(img, lab)
        img = self.photometric(img)
        return torch.from_numpy(img)[None], torch.from_numpy(lab.astype(np.int64))


# ----------------------------------------------------------------------------- training
def binary_loss(logits, target, pos_weight: float = 3.0, weight=None):
    w = torch.ones_like(target) if weight is None else weight
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.full_like(target, pos_weight), weight=w, reduction="sum") / w.sum().clamp(min=1)
    p = torch.sigmoid(logits) * w
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


LAYOUT_WEIGHTS = (0.3, 1.0, 3.0, 2.0)


def layout_loss(logits, target):
    w = torch.tensor(LAYOUT_WEIGHTS, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, target, weight=w, ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model, ds, dev, kind: str, batch_size: int = 16) -> dict:
    model.eval()
    if kind == "layout":
        conf = np.zeros((4, 4), np.int64)
        for x, y in DataLoader(ds, batch_size=batch_size):
            p = model(x.to(dev)).argmax(1).cpu().numpy()
            t = y.numpy()
            ok = t != IGNORE
            conf += np.bincount(4 * t[ok] + p[ok], minlength=16).reshape(4, 4)
        iou = [conf[c, c] / max(1, conf[c].sum() + conf[:, c].sum() - conf[c, c]) for c in range(4)]
        return {"miou": float(np.mean(iou)), **{f"iou_{n}": float(v) for n, v in zip(LAYOUT_CLASSES, iou, strict=True)}}
    tp = fp = fn = 0.0
    n_clean = n_fa = 0
    for batch in DataLoader(ds, batch_size=batch_size):
        x, y = batch[0], batch[1]
        p = torch.sigmoid(model(x.to(dev))).cpu() > 0.5
        if len(batch) > 2:  # ignored pixels count neither way
            p = p & (batch[2] > 0.5)
        t = y > 0.5
        tp += float((p & t).sum())
        fp += float((p & ~t).sum())
        fn += float((~p & t).sum())
        for k in range(len(x)):
            if not t[k].any():
                n_clean += 1
                n_fa += int(p[k].sum() > 4)
    return {"iou": tp / max(tp + fp + fn, 1), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "clean_tile_fa": n_fa / max(n_clean, 1)}


def train(
    kind: str,
    dataset: Dataset,
    out_path: str | Path,
    val_dataset: Dataset | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-3,
    base: int = 24,
    depth: int = 4,
    device: str = "auto",
    init_from: str | Path | None = None,
    log: Callable[[str], None] = print,
    pos_weight: float = 3.0,
) -> list[dict]:
    dev = pick_device(device)
    out_ch = 4 if kind == "layout" else 1
    model = (UNet.load(init_from, dev) if init_from else UNet(1, base, depth, out_channels=out_ch)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(dataset) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist, best = [], -1e9
    log(f"training {kind} UNet(base={model.base}, depth={model.depth}, out={out_ch}) on {dev}: {len(dataset)} tiles x {epochs} epochs")
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True):
            x, y = batch[0].to(dev), batch[1].to(dev)
            w = batch[2].to(dev) if len(batch) > 2 else None
            out = model(x)
            loss = layout_loss(out, y) if kind == "layout" else binary_loss(out, y, pos_weight, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(x)
            n += len(x)
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "seconds": round(time.time() - t0, 1)}
        if val_dataset is not None:
            rec.update(evaluate(model, val_dataset, dev, kind))
        hist.append(rec)
        log(json.dumps(rec))
        score = rec.get("miou", rec.get("iou", -rec["loss"]))
        if score > best:
            best = score
            model.save(out_path, {"epoch": ep + 1, "metrics": rec, "task": kind})
    out_path.with_suffix(".history.json").write_text(json.dumps(hist, indent=2))
    return hist


# ----------------------------------------------------------------------------- inference
class TiledNet:
    """Run a UNet over a whole ImageSource, writing uint8 maps into zarr arrays."""

    def __init__(self, model_path: str | Path, device: str = "auto", batch_size: int = 8):
        self.device = pick_device(device)
        self.model = UNet.load(model_path, self.device).to(self.device).eval()
        self.batch_size = batch_size
        self.layout = self.model.out_channels == 4

    @torch.no_grad()
    def predict(self, tiles: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(tiles.astype(np.float32))[:, None]
        h, w = x.shape[-2:]
        m = 2**self.model.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.model(x.to(self.device))
        if self.layout:
            return torch.softmax(y, 1).cpu().numpy()[:, :, :h, :w]
        return torch.sigmoid(y).cpu().numpy()[:, 0, :h, :w]

    def run(self, source: ImageSource, store, tile_px: int = 768, margin_px: int = 48, progress: Callable[[int, int], None] | None = None, region=None):
        """Binary nets: store[...] = uint8 probability.  Layout net: store = (class map, component prob)."""
        from chipscan.tiling import make_tiles

        tiles = make_tiles(source.shape, tile_px, margin_px, region)
        batch: list = []

        def flush():
            arr = np.stack([b[1] for b in batch])
            out = self.predict(arr)
            for (t, _), o in zip(batch, out, strict=True):
                sl = t.core_slice_in_read
                cx0, cy0, cx1, cy1 = t.core.to_int()
                if self.layout:
                    o = o[:, sl[0], sl[1]]
                    store[0][cy0:cy1, cx0:cx1] = o.argmax(0).astype(np.uint8)
                    store[1][cy0:cy1, cx0:cx1] = np.rint(o[2] * 255).astype(np.uint8)
                else:
                    store[cy0:cy1, cx0:cx1] = np.rint(o[sl] * 255).astype(np.uint8)
            batch.clear()

        for i, t in enumerate(tiles):
            x0, y0, x1, y1 = t.read.to_int()
            batch.append((t, source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0))
            if len(batch) == self.batch_size or (batch and batch[0][1].shape != batch[-1][1].shape):
                if batch[0][1].shape != batch[-1][1].shape:
                    last = batch.pop()
                    flush()
                    batch.append(last)
                else:
                    flush()
            if progress:
                progress(i + 1, len(tiles))
        if batch:
            flush()
