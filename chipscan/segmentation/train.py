"""Train the UNet on design-labelled tiles."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from chipscan.segmentation.dataset import DesignLabeledTiles
from chipscan.segmentation.unet import UNet, pick_device

console = Console()


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    p = torch.sigmoid(logits) * weight
    t = target * weight
    inter = (p * t).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1)
    return bce + 0.5 * dice.mean()


def train(
    dataset: DesignLabeledTiles,
    out_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    base: int = 24,
    depth: int = 4,
    device: str = "auto",
    val_dataset: DesignLabeledTiles | None = None,
    num_workers: int = 0,
) -> UNet:
    dev = pick_device(device)
    in_ch = 2 if dataset.use_prior else 1
    model = UNet(in_ch, base, depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dataset) // batch_size))
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    console.print(f"[bold]training[/] UNet(in={in_ch}, base={base}, depth={depth}) on {dev}, {len(dataset)} tiles x {epochs} epochs")
    best = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, w in dl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            loss = masked_bce_dice(model(x), y, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(x)
            n += len(x)
        msg = f"epoch {ep + 1}/{epochs} loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s)"
        val = None
        if val_dataset is not None:
            val = evaluate(model, val_dataset, dev, batch_size)
            msg += f" val_iou {val:.4f}"
        console.print(msg)
        score = -(val if val is not None else -(tot / max(n, 1)))
        if score < best:
            best = score
            model.save(out_path, {"epoch": ep + 1, "val_iou": val})
    return model


@torch.no_grad()
def evaluate(model: UNet, ds: DesignLabeledTiles, dev: torch.device, batch_size: int = 8) -> float:
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size)
    inter = union = 0.0
    for x, y, w in dl:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        wm = w > 0
        inter += float(((p > 0) & (y > 0) & wm).sum())
        union += float((((p > 0) | (y > 0)) & wm).sum())
    return inter / max(union, 1.0)


def split_dataset(ds: DesignLabeledTiles, val_frac: float = 0.1) -> tuple[DesignLabeledTiles, DesignLabeledTiles]:
    """Spatial split: validation tiles come from the right-most strip so they are unseen."""
    r = ds.region
    cut = r.x0 + (1 - val_frac) * r.width
    tr = DesignLabeledTiles(ds.src, ds.design, ds.warp, ds.tile_px, ds.n, ds.ignore_px, ds.use_prior, ds.augment, 1, r.__class__(r.x0, r.y0, cut, r.y1))
    va = DesignLabeledTiles(ds.src, ds.design, ds.warp, ds.tile_px, max(ds.n // 10, 16), ds.ignore_px, ds.use_prior, False, 2, r.__class__(cut, r.y0, r.x1, r.y1))
    return tr, va


def dummy_input(n: int = 1, size: int = 64, in_ch: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.rand(n, in_ch, size, size).astype(np.float32))
