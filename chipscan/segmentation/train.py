"""Train the inspection UNet on design-labelled, defect-injected tiles."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from chipscan.segmentation.dataset import InspectionTiles
from chipscan.segmentation.unet import UNet, pick_device

console = Console()


@dataclass
class EpochMetrics:
    loss: float
    metal_iou: float = 0.0
    defect_iou: float = 0.0
    defect_recall: float = 0.0
    defect_precision: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TrainHistory:
    epochs: list[dict] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.epochs, indent=2))


def _dice(p: torch.Tensor, t: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    if w is not None:
        p, t = p * w, t * w
    inter = (p * t).sum(dim=(1, 2, 3))
    return 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1)


def inspection_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, defect_pos_weight: float = 4.0) -> tuple[torch.Tensor, dict[str, float]]:
    """logits/target: (B,3,H,W) = metal, extra, missing. weight: (B,1,H,W) ignore mask for metal."""
    m_log, d_log = logits[:, :1], logits[:, 1:]
    m_t, d_t = target[:, :1], target[:, 1:]
    bce_m = F.binary_cross_entropy_with_logits(m_log, m_t, reduction="none")
    bce_m = (bce_m * weight).sum() / weight.sum().clamp_min(1.0)
    dice_m = _dice(torch.sigmoid(m_log), m_t, weight).mean()
    pw = torch.full_like(d_t, defect_pos_weight)
    bce_d = F.binary_cross_entropy_with_logits(d_log, d_t, pos_weight=pw)
    dice_d = _dice(torch.sigmoid(d_log), d_t).mean()
    loss = bce_m + 0.5 * dice_m + 2.0 * (bce_d + dice_d)
    return loss, {"metal": float(bce_m + 0.5 * dice_m), "defect": float(bce_d + dice_d)}


def train(
    dataset: InspectionTiles,
    out_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    base: int = 24,
    depth: int = 4,
    device: str = "auto",
    val_dataset: InspectionTiles | None = None,
    num_workers: int = 0,
    defect_pos_weight: float = 4.0,
    init_from: str | Path | None = None,
) -> tuple[UNet, TrainHistory]:
    dev = pick_device(device)
    in_ch = 2 if dataset.use_prior else 1
    if init_from:
        model = UNet.load(init_from, dev)
        if model.in_channels != in_ch:
            raise ValueError(f"init model expects {model.in_channels} input channels, dataset gives {in_ch}")
    else:
        model = UNet(in_ch, base, depth, out_channels=3)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(dataset) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
    console.print(f"[bold]training[/] UNet(in={in_ch}, base={model.base}, depth={model.depth}, out=3) on {dev}: {len(dataset)} tiles/epoch x {epochs} epochs, batch {batch_size}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = TrainHistory()
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, w in dl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            loss, _ = inspection_loss(model(x), y, w, defect_pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(x)
            n += len(x)
        em = EpochMetrics(tot / max(n, 1), seconds=time.time() - t0)
        if val_dataset is not None:
            v = evaluate(model, val_dataset, dev, batch_size)
            em.metal_iou, em.defect_iou, em.defect_recall, em.defect_precision = v["metal_iou"], v["defect_iou"], v["defect_recall"], v["defect_precision"]
        hist.epochs.append({"epoch": ep + 1, **em.to_dict()})
        console.print(
            f"epoch {ep + 1}/{epochs} loss {em.loss:.4f}  metal IoU {em.metal_iou:.3f}  defect IoU {em.defect_iou:.3f} "
            f"recall {em.defect_recall:.3f} precision {em.defect_precision:.3f}  ({em.seconds:.0f}s)"
        )
        score = em.defect_iou if val_dataset is not None else -em.loss
        if score > best:
            best = score
            model.save(out_path, {"epoch": ep + 1, "metrics": em.to_dict()})
    hist.save(out_path.with_suffix(".history.json"))
    return model, hist


@torch.no_grad()
def evaluate(model: UNet, ds: InspectionTiles, dev: torch.device, batch_size: int = 8, thr: float = 0.5) -> dict[str, float]:
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size)
    mi = mu = di = du = tp = fp = fn = 0.0
    for x, y, w in dl:
        p = torch.sigmoid(model(x.to(dev))).cpu()
        pm, tm, wm = p[:, 0] > thr, y[:, 0] > 0.5, w[:, 0] > 0
        mi += float((pm & tm & wm).sum())
        mu += float(((pm | tm) & wm).sum())
        pd = (p[:, 1] > thr) | (p[:, 2] > thr)
        td = (y[:, 1] > 0.5) | (y[:, 2] > 0.5)
        di += float((pd & td).sum())
        du += float((pd | td).sum())
        tp += float((pd & td).sum())
        fp += float((pd & ~td).sum())
        fn += float((~pd & td).sum())
    return {
        "metal_iou": mi / max(mu, 1.0),
        "defect_iou": di / max(du, 1.0),
        "defect_recall": tp / max(tp + fn, 1.0),
        "defect_precision": tp / max(tp + fp, 1.0),
    }


def dummy_input(n: int = 1, size: int = 64, in_ch: int = 2) -> torch.Tensor:
    return torch.from_numpy(np.random.rand(n, in_ch, size, size).astype(np.float32))
